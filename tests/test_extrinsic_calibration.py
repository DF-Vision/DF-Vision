"""
tests/test_extrinsic_calibration.py

T1 tests: recover a known transform under noise; refuse collinear
input; refuse a corrupted (units-mismatched) point; mm-vs-m guard.
Run from the project root:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_extrinsic_calibration.py -q
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "calibration"))

import numpy as np
import pytest

from compute_extrinsic_calibration import (
    kabsch, apply_transform, calibrate, CalibrationRefused,
    check_units, check_degeneracy, MIN_POINTS,
)

SEED = 20260920


def make_known_transform_points(n=12, noise_std=0.0, seed=SEED):
    rng = np.random.default_rng(seed)
    theta = 0.35
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    t_true = np.array([0.4, -0.15, 0.05])
    cam_pts = rng.uniform(-0.25, 0.25, size=(n, 3)) + np.array([0.0, 0.0, 0.5])
    base_pts = (R_true @ cam_pts.T).T + t_true
    if noise_std > 0:
        base_pts = base_pts + rng.normal(0, noise_std, size=base_pts.shape)
    return cam_pts, base_pts, R_true, t_true


def test_recovers_known_transform_noiseless():
    cam_pts, base_pts, R_true, t_true = make_known_transform_points(noise_std=0.0)
    R_est, t_est = kabsch(cam_pts, base_pts)
    assert np.allclose(R_est, R_true, atol=1e-8)
    assert np.allclose(t_est, t_true, atol=1e-8)
    assert np.isclose(np.linalg.det(R_est), 1.0, atol=1e-8)


def test_recovers_known_transform_under_noise():
    cam_pts, base_pts, R_true, t_true = make_known_transform_points(n=20, noise_std=0.002)
    ids = list(range(len(cam_pts)))
    report = calibrate(cam_pts, base_pts, ids, n_held_out=4, seed=SEED, max_residual_m=0.02)
    assert np.allclose(report["R"], R_true, atol=0.05)
    assert np.allclose(report["t"], t_true, atol=0.02)
    assert report["held_out_rms_m"] < 0.02


def test_refuses_too_few_points():
    cam_pts, base_pts, _, _ = make_known_transform_points(n=MIN_POINTS - 1)
    ids = list(range(len(cam_pts)))
    with pytest.raises(CalibrationRefused, match="minimum"):
        calibrate(cam_pts, base_pts, ids, n_held_out=1, seed=SEED, max_residual_m=0.02)


def test_refuses_collinear_input():
    n = 10
    t_vals = np.linspace(-0.3, 0.3, n)
    cam_pts = np.stack([t_vals, np.zeros(n), np.full(n, 0.5)], axis=1)  # all on a line
    base_pts = cam_pts + np.array([0.1, 0.1, 0.1])  # any translation
    ids = list(range(n))
    with pytest.raises(CalibrationRefused, match="[Dd]egenera"):
        calibrate(cam_pts, base_pts, ids, n_held_out=2, seed=SEED, max_residual_m=0.02)


def test_refuses_corrupted_point_via_bad_held_out_residual():
    cam_pts, base_pts, R_true, t_true = make_known_transform_points(n=15, noise_std=0.0)
    # Corrupt one point (simulating a bad marker detection / typo).
    base_pts = base_pts.copy()
    base_pts[-1] += np.array([0.5, 0.0, 0.0])  # 50cm error -- a real corruption, not noise
    ids = list(range(len(cam_pts)))
    # Force the corrupted point into the held-out set (deterministic via seed choice search).
    found = False
    for seed in range(200):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(cam_pts))
        if (len(cam_pts) - 1) in idx[:3]:
            found = True
            test_seed = seed
            break
    assert found, "test setup: could not find a seed placing the corrupted point in held-out"
    with pytest.raises(CalibrationRefused, match="[Hh]eld-out"):
        calibrate(cam_pts, base_pts, ids, n_held_out=3, seed=test_seed, max_residual_m=0.02)


def test_mm_vs_m_guard():
    cam_pts, base_pts, _, _ = make_known_transform_points(n=10)
    base_pts_mm_scale = base_pts * 1000.0  # someone entered millimeters as if they were meters
    with pytest.raises(CalibrationRefused, match="[Uu]nits"):
        check_units(base_pts_mm_scale, "base-frame points")


def test_units_guard_passes_normal_meters():
    cam_pts, base_pts, _, _ = make_known_transform_points(n=10)
    check_units(cam_pts, "camera-frame points")  # should not raise
    check_units(base_pts, "base-frame points")


def test_rotation_is_proper_not_reflection():
    cam_pts, base_pts, R_true, t_true = make_known_transform_points(n=10)
    R_est, _ = kabsch(cam_pts, base_pts)
    assert np.isclose(np.linalg.det(R_est), 1.0, atol=1e-6)
