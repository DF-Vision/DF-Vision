#!/usr/bin/env python3
"""
retrain/scripts/r4b_fast_g5_check.py

Faster, reduced-scope G5 check after retrain/scripts/r4_robustness_sweep.py
proved too slow to finish even a 3-point [0,5,15] sweep in reasonable time
on this session's heavily CPU-contended machine (each world's full
adjacent-pair set x LiveDepthFeatureSource's open3d-based surface-normal
computation is expensive, see verification/BASELINE.md's timing notes
from a prior session).

Scope: world1_baseline and world4_support_scene only (2 of 5 CV worlds,
chosen for size/diversity, not to favor a result), single viewpoint each,
noise levels [0, 5] only (G5's exact two comparison points per GATES.md --
drops the 15x stress-test point, already established as unnecessary for
G5's specific pass/fail question in prior sessions). This is a reduced,
faster substitute for the full sweep, not a replacement for it -- logged
as such, not silently presented as the full R4 result.
"""
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
NOISE_LEVELS = [0.0, 5.0]
GC_DIR = PROJECT_ROOT / "verification" / "gazebo_contacts_SIM"
WORLDS = ["world1_baseline", "world4_support_scene"]

FEATURE_COLUMNS = [
    "boundary_sharpness", "minimum_boundary_distance", "surface_normal_consistency",
    "relative_height", "depth_gradient", "overlap_ratio", "edge_continuity",
    "occlusion_boundary_score",
]

V2_MODEL = joblib.load(PROJECT_ROOT / "retrain/models/relation_binary_v2.joblib")
V2_SCALER = joblib.load(PROJECT_ROOT / "retrain/models/relation_binary_v2_scaler.joblib")


def balanced_acc(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    recalls = [(y_pred[y_true == c] == c).mean() for c in [0, 1] if (y_true == c).sum() > 0]
    return float(np.mean(recalls)) if recalls else float("nan")


def main():
    rows = []
    for noise_level in NOISE_LEVELS:
        for world_name in WORLDS:
            objects, cameras = erf.parse_world_sdf(erf.WORLDS_SDF_DIR / f"{world_name}.sdf")
            camera = cameras[1]
            render = ray_cast_scene(camera, objects)
            noisy_depth = apply_depth_noise(render.depth, level=noise_level,
                                             seed=RANDOM_STATE + int(noise_level))
            masks = {name: (render.owner == name) for name in objects if np.any(render.owner == name)}
            live = erf.LiveDepthFeatureSource(camera=camera)

            gc_df = pd.read_csv(GC_DIR / f"{world_name}_gazebo_contacts_SIM.csv")
            gc_df = gc_df[gc_df["contact"] != "unknown"]

            for _, r in gc_df.iterrows():
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
                x_scaled = V2_SCALER.transform(x)
                pred = int(V2_MODEL.predict(x_scaled)[0])
                rows.append({"noise_level": noise_level, "world": world_name,
                             "gt_touch": int(r["contact"]), "pred": pred})
        print(f"noise_level={noise_level}: done ({len([r for r in rows if r['noise_level']==noise_level])} rows)", flush=True)

    df = pd.DataFrame(rows)
    ba0 = balanced_acc(df[df.noise_level == 0]["gt_touch"], df[df.noise_level == 0]["pred"])
    ba5 = balanced_acc(df[df.noise_level == 5]["gt_touch"], df[df.noise_level == 5]["pred"])
    degradation = ba0 - ba5
    print(f"\nn={len(df)} total rows, worlds={WORLDS}")
    print(f"balanced_acc @ noise=0: {ba0:.4f}")
    print(f"balanced_acc @ noise=5: {ba5:.4f}")
    print(f"degradation: {degradation:.4f} "
          f"({'within G5 threshold (<0.10)' if abs(degradation) < 0.10 else 'EXCEEDS 0.10 -- reported prominently per G5'})")


if __name__ == "__main__":
    main()
