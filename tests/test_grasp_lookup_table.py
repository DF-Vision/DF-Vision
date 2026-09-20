"""
tests/test_grasp_lookup_table.py

T2 tests: synthetic transform + fake point -> expected bin -> expected
joints; out-of-reach -> IK_FAILURE; refuses without --i-am-supervising;
motor library calls mocked and asserted zero in dry-run.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "grasp"))

import numpy as np
import pytest

from lookup_table_executor import (
    LookupTable, execute_grasp_lookup_table, apply_extrinsic, nearest_bin,
    check_joint_limits, MotorInterface, IK_FAILURE,
)


def identity_extrinsic():
    return np.eye(4)


def make_table():
    return LookupTable(
        joint_names=["j1", "j2"],
        joint_limits={"j1": [-2.0, 2.0], "j2": [-2.0, 2.0]},
        max_bin_distance_m=0.05,
        bin_centers_xy=np.array([[0.2, 0.0], [0.2, 0.1], [0.3, 0.0]]),
        bin_joint_angles=[
            {"j1": 0.0, "j2": 0.1}, {"j1": 0.3, "j2": 0.1}, {"j1": 0.0, "j2": 0.5},
        ],
    )


def test_known_point_maps_to_expected_bin_and_joints():
    table = make_table()
    T = identity_extrinsic()
    grasp_point = np.array([0.2, 0.0, 0.5])  # should land exactly on bin 0
    result = execute_grasp_lookup_table(grasp_point, T, table, dry_run=True)
    assert result.bin_index == 0
    assert result.joint_angles == {"j1": 0.0, "j2": 0.1}
    assert result.pose_source == "lookup_table"
    assert result.distance_to_bin_m == pytest.approx(0.0, abs=1e-9)
    assert result.sent is False


def test_out_of_reach_returns_ik_failure():
    table = make_table()
    T = identity_extrinsic()
    grasp_point = np.array([1.5, 1.5, 0.5])  # far from every bin
    result = execute_grasp_lookup_table(grasp_point, T, table, dry_run=True)
    assert result.pose_source == "lookup_table_failed"
    assert result.failure_reason == IK_FAILURE
    assert result.joint_angles is None


def test_out_of_reach_never_extrapolates():
    """The nearest bin's joint angles must NOT be returned for an
    out-of-reach point -- only the failure record."""
    table = make_table()
    T = identity_extrinsic()
    grasp_point = np.array([0.9, 0.9, 0.5])
    result = execute_grasp_lookup_table(grasp_point, T, table, dry_run=True)
    assert result.joint_angles is None
    assert result.sent is False


def test_joint_limit_violation_is_ik_failure_not_clamped():
    table = make_table()
    table.joint_limits = {"j1": [-0.1, 0.1], "j2": [-2.0, 2.0]}  # now bin 1's j1=0.3 violates
    T = identity_extrinsic()
    grasp_point = np.array([0.2, 0.1, 0.5])  # nearest bin is bin 1
    result = execute_grasp_lookup_table(grasp_point, T, table, dry_run=True)
    assert result.pose_source == "lookup_table_failed"
    assert "IK_FAILURE" in result.failure_reason
    assert result.joint_angles is None  # never silently clamped and returned


def test_refuses_non_dry_run_without_i_am_supervising():
    table = make_table()
    T = identity_extrinsic()
    grasp_point = np.array([0.2, 0.0, 0.5])
    with pytest.raises(RuntimeError, match="i-am-supervising"):
        execute_grasp_lookup_table(grasp_point, T, table, dry_run=False, i_am_supervising=False)


def test_motor_interface_never_called_in_dry_run():
    table = make_table()
    T = identity_extrinsic()
    grasp_point = np.array([0.2, 0.0, 0.5])
    mock_motor = MagicMock(spec=MotorInterface)
    result = execute_grasp_lookup_table(grasp_point, T, table, motor=mock_motor, dry_run=True)
    mock_motor.send_joint_command.assert_not_called()
    assert result.sent is False


def test_apply_extrinsic_translates_correctly():
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    out = apply_extrinsic(T, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_nearest_bin_picks_closest():
    table = make_table()
    idx, dist = nearest_bin(table, np.array([0.21, 0.0]))
    assert idx == 0
    assert dist == pytest.approx(0.01, abs=1e-9)


def test_check_joint_limits_flags_violation():
    limits = {"j1": [-1.0, 1.0]}
    assert check_joint_limits({"j1": 2.0}, limits) is not None
    assert check_joint_limits({"j1": 0.5}, limits) is None


def test_base_class_motor_interface_raises_not_implemented():
    """Confirms there is genuinely no real hardware backend wired up --
    the base class itself refuses to send anything."""
    motor = MotorInterface()
    with pytest.raises(NotImplementedError):
        motor.send_joint_command({"j1": 0.0})
