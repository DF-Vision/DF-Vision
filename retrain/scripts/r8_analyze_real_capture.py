#!/usr/bin/env python3
"""
retrain/scripts/r8_analyze_real_capture.py

Analysis half of retrain/REAL_CAPTURE_PROTOCOL.md. Reads a real-hardware
capture folder (RGB + depth .npy pairs per staged scene, plus a
hand-filled ground_truth.csv), runs the production
LiveDepthFeatureSource -> classify_relation() -> compute_prg_edge_error()
chain, and reports accuracy/confusion matrix/prg_edge_error AUROC.

UNTESTED against real data (none exists yet in this environment -- no
camera was connected this session). Test against one dummy scene folder
with placeholder files before trusting it on a real capture session.

Masks: no real segmentation model exists for real hardware yet (per
SCHEMA.md section 5) -- this script requires the user to supply a simple
bounding box per object per scene (rgb_bbox.csv, see --help), converts it
to a bbox-rectangle mask via cv2, and sets mask_is_bbox_placeholder=True
on every resulting edge, exactly matching SCHEMA.md's documented
real-hardware fallback. Every printed result says this plainly -- per
SCHEMA.md's own rule, bbox-placeholder edges are wiring-validation only,
not a reportable accuracy number, until a real segmenter is swapped in.

In-memory config override only for relation_model_version (never edits
relation_classifier/relation_model_config.json on disk); does not touch
integration/prg_config.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))

import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import confusion_matrix

import extract_relation_features as erf
import classify_relation as cr_module
from compute_prg_edge_error import compute_prg_edge_error


def bbox_to_mask(shape, x0, y0, x1, y1):
    mask = np.zeros(shape, dtype=bool)
    mask[int(y0):int(y1), int(x0):int(x1)] = True
    return mask


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
    parser.add_argument("--capture-dir", required=True, type=str,
                         help="Folder containing scene subfolders + ground_truth.csv "
                              "(see retrain/REAL_CAPTURE_PROTOCOL.md)")
    parser.add_argument("--model-version", choices=["legacy", "v2"], default="legacy",
                         help="In-memory override only -- never written to "
                              "relation_classifier/relation_model_config.json")
    parser.add_argument("--bbox-csv", type=str, default=None,
                         help="CSV with columns scene_folder,object,x0,y0,x1,y1 "
                              "(pixel bounding boxes, one row per object per scene). "
                              "Required since no real segmenter exists yet (SCHEMA.md sec 5).")
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)
    gt_path = capture_dir / "ground_truth.csv"
    if not gt_path.exists():
        print(f"ERROR: {gt_path} not found. See retrain/REAL_CAPTURE_PROTOCOL.md for the template.")
        sys.exit(1)
    if args.bbox_csv is None or not Path(args.bbox_csv).exists():
        print("ERROR: --bbox-csv required (no real segmenter exists yet -- "
              "see SCHEMA.md section 5 for the documented bbox-placeholder fallback).")
        sys.exit(1)

    gt_df = pd.read_csv(gt_path)
    bbox_df = pd.read_csv(args.bbox_csv)

    # In-memory override of relation_model_version -- never touches the
    # config file on disk (rule: config switch stays file-based for
    # production, but this analysis script needs a CLI override for
    # comparing both versions in one session without editing the file
    # twice).
    original_loader = cr_module._load_model_version
    cr_module._load_model_version = lambda: args.model_version
    print(f"[in-memory override] relation_model_version = {args.model_version} "
          f"(relation_classifier/relation_model_config.json on disk is UNTOUCHED)")

    rows = []
    for _, gt_row in gt_df.iterrows():
        scene = gt_row["scene_folder"]
        scene_dir = capture_dir / scene
        rgb_files = sorted(scene_dir.glob("rgb_*.png"))
        depth_files = sorted(scene_dir.glob("depth_*.npy"))
        if not rgb_files or not depth_files:
            print(f"[SKIP] {scene}: no rgb_*.png / depth_*.npy found")
            continue

        obj_a, obj_b = gt_row["object_a"], gt_row["object_b"]
        scene_bboxes = bbox_df[bbox_df["scene_folder"] == scene]
        bbox_a = scene_bboxes[scene_bboxes["object"] == obj_a]
        bbox_b = scene_bboxes[scene_bboxes["object"] == obj_b]
        if bbox_a.empty or bbox_b.empty:
            print(f"[SKIP] {scene}: missing bbox for {obj_a} or {obj_b} in {args.bbox_csv}")
            continue
        bbox_a, bbox_b = bbox_a.iloc[0], bbox_b.iloc[0]

        for rgb_fp, depth_fp in zip(rgb_files, depth_files):
            rgb = cv2.imread(str(rgb_fp))
            depth = np.load(depth_fp)
            mask_a = bbox_to_mask(depth.shape, bbox_a.x0, bbox_a.y0, bbox_a.x1, bbox_a.y1)
            mask_b = bbox_to_mask(depth.shape, bbox_b.x0, bbox_b.y0, bbox_b.x1, bbox_b.y1)

            # PLACEHOLDER intrinsics -- CameraGeom derives fx/fy/cx/cy from
            # horizontal_fov (see extract_relation_features.py's CameraGeom
            # properties). 60deg (1.0472 rad) is a common RGBD-camera FOV,
            # NOT this project's actual Orbbec Gemini 2 calibration -- REPLACE
            # with the real horizontal_fov (or refactor to set fx directly)
            # before trusting any metric-scale feature (minimum_boundary_distance,
            # relative_height) from this script's output.
            camera = erf.CameraGeom(
                name="real_camera", xyz=np.zeros(3), rpy=np.zeros(3),
                width=depth.shape[1], height=depth.shape[0],
                horizontal_fov=1.0472, clip_near=0.1, clip_far=5.0,
            )

            live = erf.LiveDepthFeatureSource(camera=camera)
            feats = live.compute(depth.astype(np.float32), mask_a, mask_b)
            if any(isinstance(v, float) and np.isnan(v) for v in feats.values()):
                print(f"[SKIP] {scene} {rgb_fp.name}: NaN feature (open3d missing? masks too small?)")
                continue

            relation, confidence = cr_module.classify_relation(feats)
            edge_error = compute_prg_edge_error(
                rgb=rgb, depth=depth, target_bbox=None,
                relation=relation, confidence=confidence,
                mask_a=mask_a, mask_b=mask_b,
            )
            rows.append({
                "scene": scene, "frame": rgb_fp.name,
                "true_relation": gt_row["true_relation"],
                "pred_relation": relation, "confidence": confidence,
                "prg_edge_error": edge_error,
                "mask_is_bbox_placeholder": True,
            })

    cr_module._load_model_version = original_loader  # restore

    if not rows:
        print("No rows produced -- nothing to analyze. Check scene folders / bbox CSV.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    print(f"\n{'='*78}\nmodel_version={args.model_version} -- n={len(df)} frames "
          f"across {df['scene'].nunique()} scenes\nALL RESULTS ARE mask_is_bbox_placeholder=True "
          f"(SCHEMA.md sec 5: wiring-validation only until a real segmenter exists)\n{'='*78}")
    print(df.to_string(index=False))

    correct = df["true_relation"] == df["pred_relation"]
    print(f"\nOverall accuracy: {correct.mean():.4f} ({correct.sum()}/{len(df)})")

    labels = ["Separate", "Contact", "Support"]
    cm = confusion_matrix(df["true_relation"], df["pred_relation"], labels=labels)
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

    wrong = (~correct).astype(int)
    auc = auroc(df["prg_edge_error"], wrong)
    print(f"\nprg_edge_error AUROC for flagging wrong predictions: {auc:.4f} (G6 gate: >0.65)")
    print(f"prg_edge_error range: [{df['prg_edge_error'].min():.4f}, {df['prg_edge_error'].max():.4f}], "
          f"unique values: {df['prg_edge_error'].nunique()}")

    print("\nREMINDER: mask_is_bbox_placeholder=True on every row -- per SCHEMA.md's own")
    print("rule, this is wiring-validation only, not a reportable accuracy number, until")
    print("a real segmentation model replaces the bbox fallback.")


if __name__ == "__main__":
    main()
