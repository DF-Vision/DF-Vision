#!/usr/bin/env python3
"""
send_grasp_to_arm.py
---------------------
Minimal glue between grasp_estimator.py's output and the SO-101.

STATUS: dry-run only. Real motion is intentionally NOT implemented yet —
see the big warning in send_to_robot() below before you try to enable it.

Usage:
    python3 send_grasp_to_arm.py --grasp_x 0.24 --grasp_y 0.03 --grasp_z 2.32
    python3 send_grasp_to_arm.py --from_csv ~/grasp_results/trials_baseline_LATEST.csv --object obj_tissue_box
    python3 send_grasp_to_arm.py --grasp_x 0.24 --grasp_y 0.03 --grasp_z 2.32 --execute   # will refuse, see below
"""

import argparse
import csv
import sys


# ── Camera -> arm-base transform ─────────────────────────────────────────
# TODO (tomorrow, Phase 2 of the hardware pipeline doc): replace this with
# the real transform from your eye-to-hand calibration (Part C of the
# hardware guide). Until that calibration is done, this MUST stay identity
# — do not guess numbers here. Feeding a fake transform into a live grasp
# is worse than not having one, since it will look plausible and be wrong.
def apply_camera_to_arm_transform(x, y, z):
    """
    Input: (x, y, z) in camera-OPTICAL frame — same frame grasp_estimator.py
    now populates (x=right, y=down, z=forward from pinhole back-projection).
    Output: (x, y, z) in the SO-101's base frame.

    Currently identity — i.e. NOT calibrated. Replace the body of this
    function once Part C calibration is done.
    """
    return x, y, z


def load_grasp_from_csv(path, object_name):
    """Read the most recent row for `object_name` from a trials_*.csv."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("object_a") == object_name and row.get("grasp_x"):
                rows.append(row)
    if not rows:
        raise ValueError(f"No row with grasp_x set found for object '{object_name}' in {path}")
    latest = rows[-1]
    return float(latest["grasp_x"]), float(latest["grasp_y"]), float(latest["grasp_z"])


def send_to_robot(x, y, z, execute: bool):
    """
    Print the target pose. Only attempt real motion if execute=True —
    and even then, refuse, because the IK step doesn't exist yet.

    WHY THIS REFUSES:
    LeRobot's Robot.send_action() (see lerobot.robots.so_follower.SO101Follower)
    takes a JOINT-SPACE action — motor positions matching this robot's
    action_features — not a Cartesian (x, y, z) target. Turning a 3D point
    into joint angles requires solving inverse kinematics for the SO-101's
    specific link lengths/geometry, which is not implemented anywhere in
    this codebase. Do not stub in a fake send_action() call here — that
    would silently send an undefined/garbage action to a real motor.

    Before this function can ever move the arm for real, someone needs to
    either:
      (a) implement IK for the SO-101 (5-DOF arm + gripper) and convert
          (x, y, z) -> joint angles here, or
      (b) find and use a LeRobot-provided Cartesian control path for this
          robot, if one exists (unconfirmed as of writing this) — check
          the LeRobot docs/source directly before assuming.
    """
    print(f"[send_grasp_to_arm] Target pose (arm-base frame): "
          f"x={x:.4f} y={y:.4f} z={z:.4f}")

    if not execute:
        print("[send_grasp_to_arm] --execute not set. Dry-run only, no motion sent.")
        return

    print(
        "[send_grasp_to_arm] REFUSING to send motion: no IK solution exists "
        "yet to convert this (x, y, z) into SO-101 joint angles. Implement "
        "IK (or find a Cartesian-control path in LeRobot) before this can "
        "run live. Nothing was sent to the robot.",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp_x", type=float)
    parser.add_argument("--grasp_y", type=float)
    parser.add_argument("--grasp_z", type=float)
    parser.add_argument("--from_csv", type=str,
                         help="Path to a trials_*.csv to pull grasp_x/y/z from instead of passing them directly.")
    parser.add_argument("--object", type=str,
                         help="object_a value to look up when using --from_csv.")
    parser.add_argument("--execute", action="store_true",
                         help="Attempt real motion. Currently always refuses — see send_to_robot() docstring.")
    args = parser.parse_args()

    if args.from_csv:
        if not args.object:
            parser.error("--from_csv requires --object")
        x, y, z = load_grasp_from_csv(args.from_csv, args.object)
    elif args.grasp_x is not None and args.grasp_y is not None and args.grasp_z is not None:
        x, y, z = args.grasp_x, args.grasp_y, args.grasp_z
    else:
        parser.error("Provide either --grasp_x/--grasp_y/--grasp_z or --from_csv + --object")

    print(f"[send_grasp_to_arm] Raw grasp point (camera-optical frame): "
          f"x={x:.4f} y={y:.4f} z={z:.4f}")

    ax, ay, az = apply_camera_to_arm_transform(x, y, z)

    print(f"[send_grasp_to_arm] NOTE: camera->arm transform is currently "
          f"IDENTITY (not calibrated). Do Part C calibration before trusting "
          f"this number for anything beyond a sanity check.")

    send_to_robot(ax, ay, az, execute=args.execute)


if __name__ == "__main__":
    main()