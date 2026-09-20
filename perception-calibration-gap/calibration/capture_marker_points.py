#!/usr/bin/env python3
"""
calibration/capture_marker_points.py

Marker-pose helper for T1 (camera-to-arm extrinsic calibration). Does
NOT reimplement ArUco detection or solvePnP -- reuses
real_hardware_adapter.detect_markers()/CameraCalibration exactly as
they already exist, then writes the resulting camera-frame 3D points
(the raw solvePnP tvec, full 3D, not the yaw-only Pose simplification
that module's occlusion-specific helper produces) into
calibration/point_pairs.csv's cam_x/cam_y/cam_z columns.

Usage:
    python3 calibration/capture_marker_points.py \
        --image path/to/captured_frame.png \
        --intrinsics calibration/camera_intrinsics.yaml \
        --marker-sizes-cm 0:5.0,1:5.0,2:5.0 \
        --output calibration/point_pairs.csv

Does not touch base_x/base_y/base_z -- those are hand-measured and
filled in separately (never fabricated here).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from real_hardware_adapter import CameraCalibration, detect_markers  # noqa: E402

POINT_PAIRS_HEADER = ["marker_id", "cam_x", "cam_y", "cam_z", "base_x", "base_y", "base_z", "notes"]


def load_intrinsics(path: Path) -> CameraCalibration:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return CameraCalibration(
        fx=cfg["fx"], fy=cfg["fy"], cx=cfg["cx"], cy=cfg["cy"],
        dist_coeffs=np.array(cfg.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])),
    )


def parse_marker_sizes(spec: str) -> dict:
    """'0:5.0,1:5.0' -> {0: 0.05, 1: 0.05} (cm input, meters output)."""
    out = {}
    for pair in spec.split(","):
        mid, size_cm = pair.split(":")
        out[int(mid)] = float(size_cm) / 100.0
    return out


def capture_points(image_path: Path, calib: CameraCalibration, marker_sizes_m: dict) -> dict:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    detections = detect_markers(frame, calib, marker_sizes_m)
    points = {}
    for marker_id, det in detections.items():
        tvec = det["tvec"]
        points[marker_id] = (float(tvec[0, 0]), float(tvec[1, 0]), float(tvec[2, 0]))
    return points


def merge_into_csv(points: dict, output_path: Path) -> None:
    existing_rows = {}
    if output_path.exists():
        with open(output_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows[row["marker_id"]] = row

    for marker_id, (x, y, z) in points.items():
        key = str(marker_id)
        row = existing_rows.get(key, {h: "" for h in POINT_PAIRS_HEADER})
        row["marker_id"] = key
        row["cam_x"], row["cam_y"], row["cam_z"] = x, y, z
        row.setdefault("base_x", "REPLACE")
        row.setdefault("base_y", "REPLACE")
        row.setdefault("base_z", "REPLACE")
        existing_rows[key] = row

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POINT_PAIRS_HEADER)
        writer.writeheader()
        for row in existing_rows.values():
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--intrinsics", required=True, type=str,
                         help="YAML with fx, fy, cx, cy, dist_coeffs (from your camera calibration)")
    parser.add_argument("--marker-sizes-cm", required=True, type=str,
                         help="e.g. '0:5.0,1:5.0,2:5.0' -- marker_id:side_length_cm")
    parser.add_argument("--output", default=str(Path(__file__).parent / "point_pairs.csv"))
    args = parser.parse_args()

    calib = load_intrinsics(Path(args.intrinsics))
    marker_sizes_m = parse_marker_sizes(args.marker_sizes_cm)
    points = capture_points(Path(args.image), calib, marker_sizes_m)

    if not points:
        print(f"[warn] No markers detected in {args.image}. Nothing written.")
        sys.exit(1)

    print(f"Detected {len(points)} marker(s): {list(points.keys())}")
    merge_into_csv(points, Path(args.output))
    print(f"Updated {args.output} -- now fill in base_x/base_y/base_z by hand for each row "
          f"(measure the marker's real position in the robot base frame), then run "
          f"compute_extrinsic_calibration.py.")


if __name__ == "__main__":
    main()
