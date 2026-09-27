#!/usr/bin/env python3
"""
explore/scripts/e5_analyze_real_capture_gaps.py

Analysis half of explore/REAL_CAPTURE_PROTOCOL_V2.md. Computes geometric
prg_edge_error candidates on real captured frames and compares each
candidate's own internal gap estimate against hand-measured true gaps
(calipers/ruler), reproducing E3's criterion (c) on real data instead of
the SDF oracle.

UNTESTED against real data (none exists yet -- no camera was connected
this session, same situation as retrain/scripts/r8_analyze_real_capture.py).
In-memory use of geometric_prg_error.py / geometric_prg_error_neighbor_aware.py
only; never writes relation_classifier/prg_edge_error_config.json.

Candidates (select via --candidate, default "all"):
    c1, c2, c3   -- explore/scripts/geometric_prg_error.py (project/closure C4 fix)
    c5-neighbor-aware -- explore/scripts/geometric_prg_error_neighbor_aware.py
                          (project/closure C5; failed simulation, but this
                          harness evaluates ANY candidate the same way, per
                          project/closure-actions G3 -- it is not a claim
                          C5 is expected to do better here)

C1/C2/C3's "wrong"/error columns are computed from C1's OWN estimated_gap
(the E3-established convention: C2/C3 don't have a literal meters-scale
gap estimate of their own, only a [0,1] ambiguity score, so all three are
scored against whether C1's specific gap estimate was wrong -- unchanged
from before this file's --candidate flag was added). C5-neighbor-aware
DOES have its own native estimated_gap (unlike C2/C3), so it gets its
OWN, separate error/wrong columns computed from ITS OWN gap estimate --
scientifically the correct thing to do for a candidate that has a real
gap estimate, and it does not touch or redefine the legacy C1/C2/C3
columns in any way.
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
import geometric_prg_error_neighbor_aware as gpe_na

CANDIDATE_CHOICES = ["c1", "c2", "c3", "c5-neighbor-aware"]


def auroc(scores, labels):
    scores, labels = np.array(scores), np.array(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = np.sum(pos[:, None] > neg[None, :])
    ties = np.sum(pos[:, None] == neg[None, :])
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def _selected(args, key: str) -> bool:
    return args.candidate == "all" or args.candidate == key


def compute_rows(gt_df, bbox_df, capture_dir: Path, args) -> list:
    """Unchanged in structure/behavior from before the --candidate flag
    existed, for the c1/c2/c3 fields -- this function is the single place
    both the legacy and the new candidate's per-frame computation happen,
    but the legacy fields (C1, C2, C3, estimated_gap_m, error_m, wrong)
    are computed EXACTLY as before whenever any of c1/c2/c3/all is
    selected, byte-for-byte."""
    want_legacy = _selected(args, "c1") or _selected(args, "c2") or _selected(args, "c3")
    want_c5 = _selected(args, "c5-neighbor-aware")

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

        for depth_fp in depth_files:
            depth = np.load(depth_fp)
            row = {"scene": scene, "frame": depth_fp.name,
                   "true_relation": gt_row["true_relation"], "measured_gap_m": true_gap}
            has_any_field = False

            if want_legacy:
                c1_internal = gpe._c1_internals(depth, bbox)
                c2_score = gpe.geometric_c2_boundary_sharpness(None, depth, bbox)
                if c1_internal is None:
                    print(f"[SKIP-legacy] {scene} {depth_fp.name}: C1 internals unavailable "
                          f"(bbox too small or all-NaN depth)")
                else:
                    estimated_gap, sigma_gap, c1_score = c1_internal
                    c3_score = max(c1_score, c2_score)
                    error = abs(estimated_gap - true_gap)
                    row.update({
                        "estimated_gap_m": estimated_gap, "error_m": error,
                        "wrong": int(error > args.wrong_threshold_m),
                        "C1": c1_score, "C2": c2_score, "C3": c3_score,
                    })
                    has_any_field = True

            if want_c5:
                c5_internal = gpe_na._internals(depth, bbox)
                if c5_internal is None:
                    print(f"[SKIP-c5] {scene} {depth_fp.name}: C5-neighbor-aware internals "
                          f"unavailable (no second depth region found, or bbox too small)")
                else:
                    c5_estimated_gap, c5_sigma, c5_score = c5_internal
                    c5_error = abs(c5_estimated_gap - true_gap)
                    row.update({
                        "c5_estimated_gap_m": c5_estimated_gap, "c5_error_m": c5_error,
                        "c5_wrong": int(c5_error > args.wrong_threshold_m),
                        "C5-neighbor-aware": c5_score,
                    })
                    has_any_field = True

            if has_any_field:
                rows.append(row)

    return rows


def report(df: pd.DataFrame, args) -> None:
    print(df.to_string(index=False))

    if _selected(args, "c1") or _selected(args, "c2") or _selected(args, "c3"):
        legacy_df = df.dropna(subset=["wrong"]) if "wrong" in df.columns else df.iloc[0:0]
        if len(legacy_df) > 0:
            print(f"\nn={len(legacy_df)}, wrong fraction (error > "
                  f"{args.wrong_threshold_m*100:.0f}mm) = {legacy_df['wrong'].mean():.3f}")
            for name in ["C1", "C2", "C3"]:
                if not _selected(args, name.lower()):
                    continue
                auc = auroc(legacy_df[name], legacy_df["wrong"])
                print(f"{name}: AUROC(score, wrong) = {auc:.4f} "
                      f"(E3's simulation result for comparison: C1=0.457, C2=0.486, C3=0.463)")

    if _selected(args, "c5-neighbor-aware"):
        c5_df = df.dropna(subset=["c5_wrong"]) if "c5_wrong" in df.columns else df.iloc[0:0]
        if len(c5_df) > 0:
            print(f"\n[C5-neighbor-aware] n={len(c5_df)}, wrong fraction (own estimated_gap, "
                  f"error > {args.wrong_threshold_m*100:.0f}mm) = {c5_df['c5_wrong'].mean():.3f}")
            auc = auroc(c5_df["C5-neighbor-aware"], c5_df["c5_wrong"])
            print(f"C5-neighbor-aware: AUROC(score, own-wrong) = {auc:.4f} "
                  f"(project/closure C5's simulation result for comparison: "
                  f"criterion (a) PASS, criterion (b) FAIL 0/5 worlds -- see "
                  f"project_closure/results/c5_neighbor_aware_eval_SIM_output.txt)")

    print("\nREMINDER: real capture, but NOT citable as C1/C1-ext/C4/C4-ext evidence,")
    print("and does not validate failure prediction (needs real grasp trials).")
    print("PRG dual-signal extension (C1-ext, C4-ext, Model D) is retired for this")
    print("submission (see project_final/PRG_SIGNAL_STATUS.md) -- this script's output,")
    print("if ever run on real data, is confirmatory/appendix material only.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-dir", required=True, type=str)
    parser.add_argument("--bbox-csv", required=True, type=str)
    parser.add_argument("--scene", type=str, default=None,
                         help="Analyze only this one scene (capture-time spot check); "
                              "omit to analyze the whole capture-dir.")
    parser.add_argument("--wrong-threshold-m", type=float, default=0.01,
                         help="Matches E3's 1cm threshold by default.")
    parser.add_argument("--candidate", type=str, default="all",
                         choices=CANDIDATE_CHOICES + ["all"],
                         help="Which candidate(s) to evaluate. Default 'all' runs c1/c2/c3 "
                              "(the legacy bundle, unchanged behavior) plus c5-neighbor-aware.")
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

    rows = compute_rows(gt_df, bbox_df, capture_dir, args)

    if not rows:
        print("No rows produced -- nothing to analyze.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    report(df, args)


if __name__ == "__main__":
    main()
