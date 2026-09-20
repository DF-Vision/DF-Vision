#!/usr/bin/env python3
"""
grasp/lookup_table_executor.py

T2: config-driven joint-space lookup-table grasp execution path. NO
general IK solver -- this maps a camera-frame grasp point to the
NEAREST pre-recorded (teleop-calibrated) joint-angle bin, refusing
(IK_FAILURE) rather than extrapolating when nothing is close enough.

Pipeline:
    grasp_point_xyz_camera
        -> robot base frame, via T_base_from_camera_optical (T1's YAML output)
        -> nearest bin lookup (max_bin_distance_m, else IK_FAILURE)
        -> joint-limit check (violation -> IK_FAILURE, never clamp silently)
        -> DRY-RUN ONLY: prints the intended joint command, logs "not sent"

`pose_source` is always populated in the returned result: "lookup_table"
on success, "lookup_table_failed" on IK_FAILURE (never left blank).

HARD SAFETY RULE: this module refuses to call any real motor interface
unless `--i-am-supervising` is explicitly passed (and even then, only a
caller-supplied MotorInterface implementation could ever move anything --
none is wired to real hardware in this repository). Every code path in
this file that isn't dry-run raises unless that flag is set. This
session NEVER passes that flag.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

IK_FAILURE = "IK_FAILURE"


class MotorInterface:
    """Abstract stand-in for whatever real motor bus (LeRobot/Feetech/
    Dynamixel) eventually drives the SO-101. NOT implemented against any
    real hardware in this repository -- send_joint_command() must be
    overridden by a real backend before this could ever move anything,
    and even then only behind --i-am-supervising. Tests mock this class
    and assert send_joint_command is called ZERO times in dry-run."""

    def send_joint_command(self, joint_angles: dict) -> None:
        raise NotImplementedError(
            "No real motor backend is wired up in this repository. "
            "This is intentional -- see grasp/lookup_table_executor.py's module docstring."
        )


@dataclass
class GraspResult:
    pose_source: str
    joint_angles: Optional[dict] = None
    bin_index: Optional[int] = None
    distance_to_bin_m: Optional[float] = None
    base_frame_xyz: Optional[tuple] = None
    failure_reason: Optional[str] = None
    sent: bool = False


@dataclass
class LookupTable:
    joint_names: list
    joint_limits: dict
    max_bin_distance_m: float
    bin_centers_xy: np.ndarray  # (N, 2)
    bin_joint_angles: list      # list of dicts, one per bin

    @classmethod
    def from_yaml(cls, path: Path) -> "LookupTable":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        bins = cfg["bins"]
        centers = np.array([[b["x"], b["y"]] for b in bins])
        joint_angle_dicts = [
            dict(zip(cfg["joint_names"], b["joint_angles"])) for b in bins
        ]
        return cls(
            joint_names=cfg["joint_names"],
            joint_limits=cfg["joint_limits"],
            max_bin_distance_m=cfg["max_bin_distance_m"],
            bin_centers_xy=centers,
            bin_joint_angles=joint_angle_dicts,
        )


def apply_extrinsic(T_base_from_camera: np.ndarray, xyz_camera: np.ndarray) -> np.ndarray:
    """T_base_from_camera: 4x4 homogeneous, from T1's YAML output."""
    point_h = np.append(xyz_camera, 1.0)
    return (T_base_from_camera @ point_h)[:3]


def nearest_bin(table: LookupTable, xy_base: np.ndarray) -> tuple[Optional[int], float]:
    distances = np.linalg.norm(table.bin_centers_xy - xy_base, axis=1)
    idx = int(np.argmin(distances))
    return idx, float(distances[idx])


def check_joint_limits(joint_angles: dict, joint_limits: dict) -> Optional[str]:
    for name, value in joint_angles.items():
        lo, hi = joint_limits[name]
        if not (lo <= value <= hi):
            return f"joint {name}={value:.4f} outside limits [{lo}, {hi}]"
    return None


def execute_grasp_lookup_table(
    grasp_point_xyz_camera: np.ndarray,
    T_base_from_camera: np.ndarray,
    table: LookupTable,
    motor: Optional[MotorInterface] = None,
    dry_run: bool = True,
    i_am_supervising: bool = False,
) -> GraspResult:
    if not dry_run and not i_am_supervising:
        raise RuntimeError(
            "Refusing non-dry-run execution: --i-am-supervising was not passed. "
            "This is a hard safety rule (rule 1) -- dry-run is the only mode this "
            "session's tooling is allowed to exercise."
        )

    xyz_base = apply_extrinsic(T_base_from_camera, np.asarray(grasp_point_xyz_camera))
    bin_idx, distance = nearest_bin(table, xyz_base[:2])

    if distance > table.max_bin_distance_m:
        logger.warning(
            "Grasp point base-frame xy=%s is %.4fm from the nearest bin "
            "(max allowed %.4fm) -- IK_FAILURE, not extrapolating.",
            xyz_base[:2].tolist(), distance, table.max_bin_distance_m,
        )
        return GraspResult(
            pose_source="lookup_table_failed", bin_index=bin_idx, distance_to_bin_m=distance,
            base_frame_xyz=tuple(xyz_base.tolist()), failure_reason=IK_FAILURE,
        )

    joint_angles = table.bin_joint_angles[bin_idx]
    limit_violation = check_joint_limits(joint_angles, table.joint_limits)
    if limit_violation is not None:
        logger.warning("Joint-limit violation for bin %d: %s -- IK_FAILURE.", bin_idx, limit_violation)
        return GraspResult(
            pose_source="lookup_table_failed", bin_index=bin_idx, distance_to_bin_m=distance,
            base_frame_xyz=tuple(xyz_base.tolist()), failure_reason=f"{IK_FAILURE}: {limit_violation}",
        )

    if dry_run:
        print(f"[DRY RUN] Would send joint command: {joint_angles} (bin {bin_idx}, "
              f"distance {distance:.4f}m) -- NOT SENT.")
        logger.info("Dry-run: joint command %s NOT sent.", joint_angles)
        return GraspResult(
            pose_source="lookup_table", joint_angles=joint_angles, bin_index=bin_idx,
            distance_to_bin_m=distance, base_frame_xyz=tuple(xyz_base.tolist()), sent=False,
        )

    # Non-dry-run path -- only reachable if i_am_supervising was passed (checked above).
    if motor is None:
        raise RuntimeError("Non-dry-run execution requires a MotorInterface instance.")
    motor.send_joint_command(joint_angles)
    return GraspResult(
        pose_source="lookup_table", joint_angles=joint_angles, bin_index=bin_idx,
        distance_to_bin_m=distance, base_frame_xyz=tuple(xyz_base.tolist()), sent=True,
    )


def load_extrinsic_yaml(path: Path) -> np.ndarray:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return np.array(cfg["T_base_from_camera_optical"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grasp-point-xyz-camera", required=True, type=str,
                         help="Comma-separated x,y,z in camera frame, meters, e.g. '0.05,0.02,0.45'")
    parser.add_argument("--extrinsic", required=True, type=str,
                         help="Path to T1's T_base_from_camera_optical.yaml")
    parser.add_argument("--lookup-table", required=True, type=str)
    parser.add_argument("--i-am-supervising", action="store_true",
                         help="Required (and NEVER set by this session) for any non-dry-run execution.")
    args = parser.parse_args()

    xyz = np.array([float(v) for v in args.grasp_point_xyz_camera.split(",")])
    T = load_extrinsic_yaml(Path(args.extrinsic))
    table = LookupTable.from_yaml(Path(args.lookup_table))

    result = execute_grasp_lookup_table(
        xyz, T, table, dry_run=True, i_am_supervising=args.i_am_supervising,
    )
    print(result)


if __name__ == "__main__":
    main()
