#!/usr/bin/env python3
"""
explore/scripts/e5_analyze_real_capture_gaps.py

Analysis half of explore/REAL_CAPTURE_PROTOCOL_V2.md. Computes C1/C2/C3
geometric prg_edge_error candidates on real captured frames and compares
C1's internal gap estimate against hand-measured true gaps (calipers/
ruler), reproducing E3's criterion (c) on real data instead of the SDF
oracle.

UNTESTED against real data (none exists yet -- no camera was connected
this session, same situation as retrain/scripts/r8_analyze_real_capture.py).
In-memory use of geometric_prg_error.py only; never writes
relation_classifier/prg_edge_error_config.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "explore" / "scripts"))

import numpy as np
import pandas as pd

import geometric_prg_error as gpe


def auroc(scores, labels):
    scores, labels = np.array(scores), np.array(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = np.sum(pos[:, None] > neg[None, :])
    ties = np.sum(pos[:, None] == neg[None, :])
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=str)
    parser.add_argument("--bbox-csv", required=True, type=str)
    parser.add_argument("--scene", type=str, default=None,
                         help="Analyze only this one scene (capture-time spot check); "
                              "omit to analyze the whole capture-dir.")
    parser.add_argument("--wrong-threshold-m", type=float, default=0.01,
                         help="Matches E3's 1cm threshold by default.")
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)
    gt_path = capture_dir / "ground_truth.csv"
    if not gt_path.exists():
        print(f"ERROR: {gt_path} not found. See explore/REAL_CAPTURE_PROTOCOL_V2.md.")
        sys.exit(1)
    gt_df = pd.read_csv(gt_path)
    bbox_df = pd.read_csv(args.bbox_csv)

    if args.scene:
        gt_df = gt_df[gt_df["scene_folder"] == args.scene]
        if gt_df.empty:
            print(f"ERROR: scene {args.scene!r} not found in {gt_path}")
            sys.exit(1)

    rows = []
    for _, gt_row in gt_df.iterrows():
        scene = gt_row["scene_folder"]
        scene_dir = capture_dir / scene
        rgb_files = sorted(scene_dir.glob("rgb_*.png"))
        depth_files = sorted(scene_dir.glob("depth_*.npy"))
        if not rgb_files or not depth_files:
            print(f"[SKIP] {scene}: no rgb_*.png / depth_*.npy found")
            continue

        obj_a = gt_row["object_a"]
        scene_bbox = bbox_df[(bbox_df["scene_folder"] == scene) & (bbox_df["object"] == obj_a)]
        if scene_bbox.empty:
            print(f"[SKIP] {scene}: missing bbox for {obj_a} in {args.bbox_csv}")
            continue
        bbox_row = scene_bbox.iloc[0]
        bbox = (bbox_row.x0, bbox_row.y0, bbox_row.x1, bbox_row.y1)
        true_gap = float(gt_row["measured_gap_m"])

        import cv2
        for depth_fp in depth_files:
            depth = np.load(depth_fp)
            c1_internal = gpe._c1_internals(depth, bbox)
            c2_score = gpe.geometric_c2_boundary_sharpness(None, depth, bbox)
            if c1_internal is None:
                print(f"[SKIP] {scene} {depth_fp.name}: C1 internals unavailable "
                      f"(bbox too small or all-NaN depth)")
                continue
            estimated_gap, sigma_gap, c1_score = c1_internal
            c3_score = max(c1_score, c2_score)
            error = abs(estimated_gap - true_gap)
            rows.append({
                "scene": scene, "frame": depth_fp.name,
                "true_relation": gt_row["true_relation"], "measured_gap_m": true_gap,
                "estimated_gap_m": estimated_gap, "error_m": error,
                "wrong": int(error > args.wrong_threshold_m),
                "C1": c1_score, "C2": c2_score, "C3": c3_score,
            })

    if not rows:
        print("No rows produced -- nothing to analyze.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nn={len(df)}, wrong fraction (error > {args.wrong_threshold_m*100:.0f}mm) = {df['wrong'].mean():.3f}")
    for name in ["C1", "C2", "C3"]:
        auc = auroc(df[name], df["wrong"])
        print(f"{name}: AUROC(score, wrong) = {auc:.4f} "
              f"(E3's simulation result for comparison: C1=0.457, C2=0.486, C3=0.463)")

    print("\nREMINDER: real capture, but NOT citable as C1/C1-ext/C4/C4-ext evidence,")
    print("and does not validate failure prediction (needs real grasp trials).")


if __name__ == "__main__":
    main()
