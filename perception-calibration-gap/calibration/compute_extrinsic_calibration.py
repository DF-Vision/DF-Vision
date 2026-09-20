#!/usr/bin/env python3
"""
calibration/compute_extrinsic_calibration.py

Camera-to-arm extrinsic calibration (T1, hard gate for any real grasp).

Solves the rigid transform T_base_from_camera_optical such that, for
each marker i:
    p_base_i approx= R @ p_camera_i + t
via the Kabsch/SVD algorithm (closed-form least-squares rigid alignment,
Umeyama/Kabsch 1976/1991), with a proper-rotation check (det(R) == +1;
if SVD gives a reflection, the standard sign-flip correction on the
smallest singular vector is applied).

Input: one CSV with per-marker point pairs (see
`calibration/point_pairs_template.csv`, written on first run if missing)
with columns:
    marker_id, cam_x, cam_y, cam_z, base_x, base_y, base_z
cam_* comes from `calibration/capture_marker_points.py` (solvePnP via
the ALREADY-EXISTING `real_hardware_adapter.detect_markers`, not
reimplemented here). base_* is HAND-MEASURED by a human with a ruler/tape
measure in the robot's base frame and filled in by hand -- this script
never fabricates or guesses those values.

REFUSES to save a transform (still prints a report either way) if:
  - fewer than MIN_POINTS (6) pairs are provided,
  - the points are geometrically degenerate (near-collinear/near-planar:
    smallest/largest singular value ratio of the camera-frame points'
    centered covariance < DEGENERACY_RATIO_THRESHOLD),
  - any point's magnitude exceeds UNIT_GUARD_MAX_M (a units mistake --
    millimeters entered as meters),
  - held-out RMS residual exceeds --max-residual-m (default 2cm).

Output: YAML with a 4x4 homogeneous `T_base_from_camera_optical` matrix
(meters), plus a residual report (train/held-out RMS and max, per-point
errors).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

MIN_POINTS = 6
MIN_HELD_OUT = 2
DEGENERACY_RATIO_THRESHOLD = 0.01
UNIT_GUARD_MAX_M = 5.0
DEFAULT_MAX_RESIDUAL_M = 0.02

POINT_PAIRS_TEMPLATE = Path(__file__).parent / "point_pairs_template.csv"
POINT_PAIRS_HEADER = ["marker_id", "cam_x", "cam_y", "cam_z", "base_x", "base_y", "base_z", "notes"]


class CalibrationRefused(Exception):
    """Raised whenever this script refuses to save a transform. Callers
    (and the CLI) must treat this as a hard stop, never a warning to
    route around."""


def write_template_if_missing(path: Path) -> None:
    if path.exists():
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(POINT_PAIRS_HEADER)
        writer.writerow([0, 0.10, 0.05, 0.45, "REPLACE", "REPLACE", "REPLACE",
                          "marker 0 -- camera-frame point from capture_marker_points.py, "
                          "base-frame point measured by hand"])
        writer.writerow([1, 0.15, -0.05, 0.42, "REPLACE", "REPLACE", "REPLACE", ""])
    print(f"[info] wrote starter template to {path} -- fill in real measured base_x/y/z "
          f"and re-run.")


def load_point_pairs(path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    cam_pts, base_pts, ids = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cx, cy, cz = float(row["cam_x"]), float(row["cam_y"]), float(row["cam_z"])
                bx, by, bz = float(row["base_x"]), float(row["base_y"]), float(row["base_z"])
            except (ValueError, KeyError):
                continue  # unfilled/placeholder row ("REPLACE" etc.) -- skip, don't guess
            cam_pts.append([cx, cy, cz])
            base_pts.append([bx, by, bz])
            ids.append(row.get("marker_id", ""))
    return np.array(cam_pts), np.array(base_pts), ids


def check_units(points: np.ndarray, label: str) -> None:
    if points.size == 0:
        return
    max_mag = float(np.max(np.linalg.norm(points, axis=1)))
    if max_mag > UNIT_GUARD_MAX_M:
        raise CalibrationRefused(
            f"Units guard: {label} contains a point with magnitude {max_mag:.3f} "
            f"(> {UNIT_GUARD_MAX_M}m). A robot workspace / tabletop camera view is "
            f"well under {UNIT_GUARD_MAX_M}m -- this looks like millimeters entered "
            f"as meters. Refusing rather than silently computing a 1000x-wrong "
            f"transform."
        )


def check_degeneracy(points: np.ndarray) -> None:
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered
    singular_values = np.linalg.svd(cov, compute_uv=False)
    ratio = float(singular_values.min() / max(singular_values.max(), 1e-12))
    if ratio < DEGENERACY_RATIO_THRESHOLD:
        raise CalibrationRefused(
            f"Degeneracy guard: camera-frame points' smallest/largest singular-value "
            f"ratio is {ratio:.6f} (< {DEGENERACY_RATIO_THRESHOLD}) -- points are "
            f"near-collinear or near-planar and do not sufficiently constrain a full "
            f"3D rigid rotation. Capture points spanning more of the workspace volume "
            f"(not all on one line or one plane) and re-run."
        )


def kabsch(cam_pts: np.ndarray, base_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (R, t) such that base_pts ~= cam_pts @ R.T + t, via SVD.
    Standard Kabsch algorithm with the reflection-correction step."""
    cam_centroid = cam_pts.mean(axis=0)
    base_centroid = base_pts.mean(axis=0)
    cam_c = cam_pts - cam_centroid
    base_c = base_pts - base_centroid

    H = cam_c.T @ base_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    t = base_centroid - R @ cam_centroid
    return R, t


def apply_transform(R: np.ndarray, t: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (R @ points.T).T + t


def rms_and_max_residual(predicted: np.ndarray, actual: np.ndarray) -> tuple[float, float, np.ndarray]:
    errors = np.linalg.norm(predicted - actual, axis=1)
    return float(np.sqrt(np.mean(errors ** 2))), float(errors.max()), errors


def calibrate(cam_pts: np.ndarray, base_pts: np.ndarray, ids: list,
              n_held_out: int, seed: int, max_residual_m: float) -> dict:
    n = len(cam_pts)
    if n < MIN_POINTS:
        raise CalibrationRefused(
            f"Only {n} point pairs provided (< {MIN_POINTS} minimum). A 6-DOF rigid "
            f"transform needs at least {MIN_POINTS} well-spread correspondences to be "
            f"numerically stable under real measurement noise. Capture more markers."
        )
    if n - n_held_out < MIN_POINTS - MIN_HELD_OUT:
        raise CalibrationRefused(
            f"{n} pairs with {n_held_out} held out leaves too few for a stable fit. "
            f"Provide more points or reduce --held-out."
        )

    check_units(cam_pts, "camera-frame points")
    check_units(base_pts, "base-frame points")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    held_idx, train_idx = idx[:n_held_out], idx[n_held_out:]

    check_degeneracy(cam_pts[train_idx])

    R, t = kabsch(cam_pts[train_idx], base_pts[train_idx])
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
        raise CalibrationRefused(
            f"Computed rotation has det(R)={np.linalg.det(R):.6f}, not +1 -- this "
            f"should be impossible given the Kabsch reflection correction; refusing "
            f"rather than saving a non-physical (reflection/scaling) transform. "
            f"This indicates a bug, not bad input -- investigate before proceeding."
        )

    train_pred = apply_transform(R, t, cam_pts[train_idx])
    train_rms, train_max, train_errors = rms_and_max_residual(train_pred, base_pts[train_idx])

    held_pred = apply_transform(R, t, cam_pts[held_idx])
    held_rms, held_max, held_errors = rms_and_max_residual(held_pred, base_pts[held_idx])

    result = {
        "R": R, "t": t, "n_points": n, "n_train": len(train_idx), "n_held_out": len(held_idx),
        "train_rms_m": train_rms, "train_max_m": train_max,
        "held_out_rms_m": held_rms, "held_out_max_m": held_max,
        "held_out_errors_m": held_errors.tolist(),
        "train_ids": [ids[i] for i in train_idx], "held_out_ids": [ids[i] for i in held_idx],
    }

    if held_rms > max_residual_m:
        raise CalibrationRefused(
            f"Held-out RMS residual {held_rms:.4f}m exceeds tolerance {max_residual_m}m. "
            f"REFUSING TO SAVE. Report (for diagnosis): {result}"
        )

    return result


def to_homogeneous_yaml(R: np.ndarray, t: np.ndarray, report: dict) -> dict:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return {
        "T_base_from_camera_optical": T.tolist(),
        "units": "meters",
        "residuals": {
            "train_rms_m": report["train_rms_m"], "train_max_m": report["train_max_m"],
            "held_out_rms_m": report["held_out_rms_m"], "held_out_max_m": report["held_out_max_m"],
            "n_train": report["n_train"], "n_held_out": report["n_held_out"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", default=str(POINT_PAIRS_TEMPLATE.parent / "point_pairs.csv"),
                         help="Path to the point-pairs CSV (created if missing)")
    parser.add_argument("--output", default=str(POINT_PAIRS_TEMPLATE.parent / "T_base_from_camera_optical.yaml"))
    parser.add_argument("--held-out", type=int, default=MIN_HELD_OUT)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--max-residual-m", type=float, default=DEFAULT_MAX_RESIDUAL_M)
    args = parser.parse_args()

    points_path = Path(args.points)
    write_template_if_missing(points_path)
    cam_pts, base_pts, ids = load_point_pairs(points_path)

    print(f"Loaded {len(cam_pts)} filled-in point pairs from {points_path}")
    if len(cam_pts) == 0:
        print("[error] No filled-in point pairs (all rows are still placeholders). "
              "Capture marker points and measure base-frame points, then re-run.")
        sys.exit(1)

    try:
        report = calibrate(cam_pts, base_pts, ids, args.held_out, args.seed, args.max_residual_m)
    except CalibrationRefused as e:
        print(f"[REFUSED] {e}")
        sys.exit(1)

    print(f"Train RMS={report['train_rms_m']:.4f}m max={report['train_max_m']:.4f}m "
          f"(n={report['n_train']})")
    print(f"Held-out RMS={report['held_out_rms_m']:.4f}m max={report['held_out_max_m']:.4f}m "
          f"(n={report['n_held_out']}) -- tolerance {args.max_residual_m}m -- PASS")

    out = to_homogeneous_yaml(report["R"], report["t"], report)
    with open(args.output, "w") as f:
        yaml.safe_dump(out, f, default_flow_style=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
