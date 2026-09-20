"""
e2e_integration_rehearsal_SYNTHETIC.py -- SYNTHETIC - NOT A RESULT

C8: chains replay frames -> Gap -> PRG (stub) -> calibration transform
(synthetic known, fitted by the REAL solver) -> lookup-table dry-run ->
logger -> data_quality_report -> Gap-only stats -> conformal gate, using
ONLY code this session verified/built (C1-C7) -- no stage reimplemented.
A sibling to daily_integration_check.py (not folded into it, so that
existing, already-green check is never put at risk). See
project_closure/CRITERIA.md's C8 section for the pre-committed design.

Nothing here is real trial data or a real-hardware result.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "real_camera"))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "calibration"))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "grasp"))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "gazebo"))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "mock-pipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "analysis"))

import capture_and_gap
from replay_source import ReplayFrameSource
import generate_replay_fixture
from compute_extrinsic_calibration import calibrate
from lookup_table_executor import (
    LookupTable, execute_grasp_lookup_table, apply_extrinsic, MotorInterface,
)
from unittest.mock import MagicMock
from logger import TrialLogger, REQUIRED_FIELDS
from data_quality_report import (
    session_object_confounding_check, controller_freeze_check, prg_stub_vs_real_counts,
)
from mock_data_generator import generate_mock_trials
from analysis_c1 import fit_models
from conformal_gate import ConformalGate, three_split

RANDOM_STATE = 20260921


def log(s=""):
    print(s)


def stage_replay_to_gap(tmp_dir: Path):
    log("\n--- Stage 1: replay frames -> Gap (+ PRG stub) ---")
    fixture_dir = tmp_dir / "e2e_fixture_SYNTHETIC"
    generate_replay_fixture.generate(str(fixture_dir), seed=RANDOM_STATE)
    log_path = tmp_dir / "e2e_replay_SYNTHETIC.jsonl"
    trials = capture_and_gap.run_replay(str(fixture_dir), log_path)
    good_trials = [t for t in trials if t.trial_id < 4]  # the 4 non-faulted fixture trials
    assert good_trials, "expected at least one successfully replayed trial"
    for t in good_trials:
        assert t.prg_edges is not None
        assert t.prg_is_stub is True, "use_stub stays true -- PRG must come from the stub path"
    log(f"  {len(good_trials)} good trials, all prg_is_stub=True (stub path, as configured)")
    return good_trials, log_path


def stage_calibration():
    log("\n--- Stage 2: calibration transform (synthetic known, fitted by the real solver) ---")
    rng = np.random.default_rng(RANDOM_STATE)
    theta = 0.3  # a nontrivial rotation, radians, about the z axis
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    t_true = np.array([0.15, -0.08, 0.42])

    n_points = 12
    cam_pts = rng.uniform(-0.3, 0.3, size=(n_points, 3)) + np.array([0, 0, 0.5])
    base_pts = (R_true @ cam_pts.T).T + t_true
    ids = list(range(n_points))

    result = calibrate(cam_pts, base_pts, ids, n_held_out=3, seed=RANDOM_STATE, max_residual_m=0.01)
    assert result["held_out_rms_m"] < 1e-6, f"expected near-zero residual on noiseless synthetic data, got {result['held_out_rms_m']}"
    log(f"  Recovered transform: held-out RMS residual={result['held_out_rms_m']:.2e}m (near-zero, as expected)")

    T = np.eye(4)
    T[:3, :3] = result["R"]
    T[:3, 3] = result["t"]
    return T


def stage_lookup_table(T_base_from_camera: np.ndarray, grasp_point_xyz_camera: np.ndarray):
    log("\n--- Stage 3: lookup-table dry-run ---")
    table = LookupTable(
        joint_names=["j1", "j2"],
        joint_limits={"j1": [-3.0, 3.0], "j2": [-3.0, 3.0]},
        max_bin_distance_m=0.5,  # generous -- this rehearsal isn't testing IK_FAILURE
        bin_centers_xy=np.array([[0.1, -0.1], [0.15, -0.08], [0.2, -0.05]]),
        bin_joint_angles=[{"j1": 0.0, "j2": 0.1}, {"j1": 0.1, "j2": 0.2}, {"j1": 0.2, "j2": 0.3}],
    )
    mock_motor = MagicMock(spec=MotorInterface)
    result = execute_grasp_lookup_table(
        grasp_point_xyz_camera, T_base_from_camera, table,
        motor=mock_motor, dry_run=True, i_am_supervising=False,
    )
    mock_motor.send_joint_command.assert_not_called()
    assert result.sent is False
    log(f"  pose_source={result.pose_source}, sent={result.sent} (NOT SENT, dry-run only)")
    log(f"  Motor interface call count: {mock_motor.send_joint_command.call_count} (must be 0)")
    return result


def stage_logger(good_trials, base_log_path: Path, controller_hash: str):
    log("\n--- Stage 4: logger (schema-correct rows) ---")
    for t in good_trials:
        missing = [f for f in REQUIRED_FIELDS if getattr(t, f, None) is None]
        assert not missing, f"trial {t.trial_id} missing required fields: {missing}"
        t.controller_git_hash = controller_hash
    log(f"  {len(good_trials)} rows schema-correct (all REQUIRED_FIELDS populated); "
        f"controller_git_hash={controller_hash} on every row")
    return good_trials


def stage_data_quality(good_trials, tmp_dir: Path):
    log("\n--- Stage 5: data_quality_report (incl. controller-freeze check) ---")
    import pandas as pd
    df = pd.DataFrame([vars(t) for t in good_trials])
    df["session_id"] = 0  # all from one replay session -- confounding check should flag this (1 object)

    confound = session_object_confounding_check(df)
    log(f"  Confounding check: {confound}")

    stub_counts = prg_stub_vs_real_counts([vars(t) for t in good_trials])
    log(f"  PRG stub/real counts: {stub_counts}")
    assert stub_counts["n_real_prg_rows"] == 0, "stub path must produce zero REAL prg rows"

    freeze_ok = controller_freeze_check(df, str(tmp_dir / "e2e_SYNTHETIC_log_ok.jsonl"))
    log(f"  Controller-freeze check (single hash): {freeze_ok}")
    assert freeze_ok["ok"] is True

    df_changed = df.copy()
    df_changed.loc[df_changed.index[-1], "controller_git_hash"] = "deliberately_different_hash_SYNTHETIC"
    freeze_fail = controller_freeze_check(df_changed, str(tmp_dir / "e2e_SYNTHETIC_log_changed.jsonl"))
    log(f"  Controller-freeze check (deliberate mid-dataset change, no tag): {freeze_fail}")
    assert freeze_fail["ok"] is False, "freeze check must FAIL on an untagged mid-dataset controller change"
    log("  Freeze check correctly FAILS on the deliberate change -- not just present, demonstrably fires.")


def stage_gap_only_stats():
    log("\n--- Stage 6: Gap-only stats (PRG columns absent) ---")
    df = generate_mock_trials(seed=RANDOM_STATE, gap_signal_multiplier=1.0)
    assert "prg_edge_error" not in df.columns and "prg_edges" not in df.columns, (
        "Gap-only synthetic batch must have no prg columns at all"
    )
    models = fit_models(df)
    log(f"  Model B fit on {len(df)} SYNTHETIC rows (larger batch -- the E2E replay fixture alone "
        f"is too small to fit a logistic model, stated explicitly, not silently swapped in). "
        f"Gap_initial coef={models['B'].params['Gap_initial']:.3f}")
    return models


def stage_conformal_gate():
    log("\n--- Stage 7: conformal gate ---")
    df = generate_mock_trials(seed=RANDOM_STATE + 1, gap_signal_multiplier=1.0)
    calib_df, test_df = three_split(df, calib_frac=0.7, seed=RANDOM_STATE)
    gate = ConformalGate(alpha=0.10)
    gate.fit(calib_df)
    coverage = gate.empirical_coverage(test_df)
    log(f"  Pooled coverage={coverage['pooled_coverage']:.3f}, flag_rate={coverage['pooled_flag_rate']:.3f}")
    return coverage


def main():
    log("SYNTHETIC - NOT A RESULT")
    log(f"Generated: {datetime.now(timezone.utc).isoformat()}")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        good_trials, log_path = stage_replay_to_gap(tmp_dir)
        T = stage_calibration()
        grasp_point_xyz_camera = np.array([0.02, -0.01, 0.5])  # synthetic camera-frame target point
        stage_lookup_table(T, grasp_point_xyz_camera)
        good_trials = stage_logger(good_trials, log_path, controller_hash="e2e_SYNTHETIC_controller_v1")
        stage_data_quality(good_trials, tmp_dir)
        stage_gap_only_stats()
        stage_conformal_gate()

    log(f"\n{'='*78}")
    log("E2E INTEGRATION REHEARSAL: PASS (all stages completed, all assertions held)")
    log("SYNTHETIC. Not real trial data. Not a real-hardware result.")
    log(f"{'='*78}")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
