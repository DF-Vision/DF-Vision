#!/usr/bin/env python3
"""
retrain/scripts/r4_robustness_sweep.py

R4: robustness / distribution-shift sweep (SIM). Sweeps depth noise level
0x-15x (retrain.scripts.noise_model) on the LEGACY frozen model and on
v2 (once trained by R3), using the real gazebo_contact-labeled pairs
(verification/gazebo_contacts_SIM/) as ground truth, on world1-5 only
(world6_dense_mixed stays untouched until R5).

Also reports per-feature drift vs each model's own scaler stats at each
noise level, and demonstrates the out-of-distribution WARNING behavior
this session adds to the classifier wrapper (R6) by replicating its
>4 SD check standalone here first, so the R4 report and the R6 code
change are built from the same, already-verified logic.

Does not modify any production file. All output SIM-labeled.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "retrain" / "scripts"))

import numpy as np
import pandas as pd
import joblib

import extract_relation_features as erf
from validate_live_depth_feature_source import ray_cast_scene
from noise_model import apply_depth_noise

RANDOM_STATE = 20260920
NOISE_LEVELS = [0.0, 5.0, 15.0]  # reduced from the full [0,1,2,3,5,10,15] sweep
# after the first level alone took ~8-13 min of wall time on this heavily
# CPU-contended machine (see retrain/BASELINE.md) -- these 3 points cover
# GATES.md's G5 comparison (0 vs 5, "highest realistic level") plus the
# 15x stress-test endpoint used in prior sessions, without the ~50min the
# full 7-point sweep would cost for marginal extra resolution given G1/G2
# already fail decisively regardless (see retrain/models/r3_training_summary.txt).
GC_DIR = PROJECT_ROOT / "verification" / "gazebo_contacts_SIM"
OUT_CSV = PROJECT_ROOT / "retrain" / "r4_robustness_sweep_SIM.csv"

WORLDS = [
    "world1_baseline", "world2_dense_clutter", "world3_thin_objects",
    "world4_support_scene", "world5_occlusion_scene",
]  # world6 excluded -- held out per GATES.md

FEATURE_COLUMNS = [
    "boundary_sharpness", "minimum_boundary_distance", "surface_normal_consistency",
    "relative_height", "depth_gradient", "overlap_ratio", "edge_continuity",
    "occlusion_boundary_score",
]

LEGACY_MODEL = joblib.load(PROJECT_ROOT / "relation_classifier/models/logistic_model.joblib")
LEGACY_SCALER = joblib.load(PROJECT_ROOT / "relation_classifier/models/scaler.joblib")

V2_MODEL_PATH = PROJECT_ROOT / "retrain/models/relation_binary_v2.joblib"
V2_SCALER_PATH = PROJECT_ROOT / "retrain/models/relation_binary_v2_scaler.joblib"


def load_gc():
    frames = []
    for w in WORLDS:
        d = pd.read_csv(GC_DIR / f"{w}_gazebo_contacts_SIM.csv")
        d["world_id"] = w
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    return df[df["contact"] != "unknown"].copy()


def balanced_acc(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    recalls = []
    for cls in [0, 1]:
        mask = y_true == cls
        if mask.sum() > 0:
            recalls.append((y_pred[mask] == cls).mean())
    return float(np.mean(recalls)) if recalls else float("nan")


def main():
    gc_df = load_gc()
    v2_available = V2_MODEL_PATH.exists() and V2_SCALER_PATH.exists()
    if v2_available:
        v2_model = joblib.load(V2_MODEL_PATH)
        v2_scaler = joblib.load(V2_SCALER_PATH)
        print(f"v2 model found at {V2_MODEL_PATH}, including in sweep.")
    else:
        print(f"No v2 model at {V2_MODEL_PATH} yet -- sweeping LEGACY only.")

    rows = []
    for noise_level in NOISE_LEVELS:
        for world_name in WORLDS:
            objects, cameras = erf.parse_world_sdf(erf.WORLDS_SDF_DIR / f"{world_name}.sdf")
            camera = cameras[1]
            render = ray_cast_scene(camera, objects)
            seed = RANDOM_STATE + int(noise_level * 1000) + hash(world_name) % 1000
            noisy_depth = apply_depth_noise(render.depth, level=noise_level, seed=seed)
            masks = {name: (render.owner == name) for name in objects if np.any(render.owner == name)}
            live = erf.LiveDepthFeatureSource(camera=camera)

            world_gc = gc_df[gc_df["world_id"] == world_name]
            for _, r in world_gc.iterrows():
                a, b = r["object_a"], r["object_b"]
                if a not in masks or b not in masks:
                    continue
                mask_a, mask_b = masks[a], masks[b]
                if not np.any(mask_a) or not np.any(mask_b):
                    continue
                feats = live.compute(noisy_depth, mask_a, mask_b)
                if any(isinstance(v, float) and np.isnan(v) for v in feats.values()):
                    continue
                x = np.array([[feats[c] for c in FEATURE_COLUMNS]])
                gt_touch = int(r["contact"])

                x_legacy = LEGACY_SCALER.transform(x)
                pred_legacy = int(LEGACY_MODEL.predict(x_legacy)[0])
                z_legacy = (x[0] - LEGACY_SCALER.mean_) / LEGACY_SCALER.scale_
                max_abs_z_legacy = float(np.max(np.abs(z_legacy)))

                row = {
                    "noise_level": noise_level, "world": world_name, "pair": f"{a}/{b}",
                    "gt_touch": gt_touch, "pred_legacy": pred_legacy,
                    "max_abs_z_legacy": max_abs_z_legacy,
                    "ood_warning_legacy": max_abs_z_legacy > 4.0,
                }
                if v2_available:
                    x_v2 = v2_scaler.transform(x)
                    pred_v2 = int(v2_model.predict(x_v2)[0])
                    z_v2 = (x[0] - v2_scaler.mean_) / v2_scaler.scale_
                    max_abs_z_v2 = float(np.max(np.abs(z_v2)))
                    row["pred_v2"] = pred_v2
                    row["max_abs_z_v2"] = max_abs_z_v2
                    row["ood_warning_v2"] = max_abs_z_v2 > 4.0
                rows.append(row)

        print(f"noise_level={noise_level}: done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows -> {OUT_CSV}\n")

    print(f"{'noise_level':<12s} {'bal_acc_legacy':<16s} {'ood_warn_rate_legacy':<20s}", end="")
    if v2_available:
        print(f"{'bal_acc_v2':<12s} {'ood_warn_rate_v2':<16s}")
    else:
        print()
    for nl in NOISE_LEVELS:
        sub = df[df["noise_level"] == nl]
        ba_legacy = balanced_acc(sub["gt_touch"], sub["pred_legacy"])
        ood_legacy = sub["ood_warning_legacy"].mean()
        line = f"{nl:<12.1f} {ba_legacy:<16.4f} {ood_legacy:<20.4f}"
        if v2_available:
            ba_v2 = balanced_acc(sub["gt_touch"], sub["pred_v2"])
            ood_v2 = sub["ood_warning_v2"].mean()
            line += f"{ba_v2:<12.4f} {ood_v2:<16.4f}"
        print(line)

    if v2_available:
        ba_at_0 = balanced_acc(df[df.noise_level == 0]["gt_touch"], df[df.noise_level == 0]["pred_v2"])
        ba_at_5 = balanced_acc(df[df.noise_level == 5]["gt_touch"], df[df.noise_level == 5]["pred_v2"])
        degradation = ba_at_0 - ba_at_5
        print(f"\nG5 check: v2 balanced accuracy at noise=0: {ba_at_0:.4f}, at noise=5 (GATES.md's "
              f"'highest realistic level'): {ba_at_5:.4f}, degradation={degradation:.4f} "
              f"({'PASS, <0.10' if degradation < 0.10 else 'FAIL, >=0.10 -- reported prominently per G5'})")


if __name__ == "__main__":
    main()
