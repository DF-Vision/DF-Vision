#!/usr/bin/env python3
"""
diagnosis/scripts/run_binary_model_diagnostic.py

Overnight diagnostic for PROJECT_STATUS.md section 1 (binary touching-model
"always Separate" finding). Reuses existing, validated infrastructure only:
  - extract_relation_features.parse_world_sdf / LiveDepthFeatureSource /
    enumerate_adjacent_pairs / SDFGeometryRelationSource (Krupa's code,
    untouched)
  - validate_live_depth_feature_source.ray_cast_scene (this repo's
    from-scratch analytic ray-caster, validated 182/0/18 last session,
    untouched)

Does NOT retrain or modify any model/scaler. Read-only diagnostic.
Every number printed here is SYNTHETIC (ray-cast depth on staged SDF
scenes) -- not evidence for any paper claim.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))

import numpy as np
import joblib

import extract_relation_features as erf
from validate_live_depth_feature_source import ray_cast_scene, HAND_PICKED_PAIRS

FEATURE_COLUMNS = [
    "boundary_sharpness", "minimum_boundary_distance", "surface_normal_consistency",
    "relative_height", "depth_gradient", "overlap_ratio", "edge_continuity",
    "occlusion_boundary_score",
]

MODEL = joblib.load(PROJECT_ROOT / "relation_classifier/models/logistic_model.joblib")
SCALER = joblib.load(PROJECT_ROOT / "relation_classifier/models/scaler.joblib")


def zscore_row(feature_dict):
    x = np.array([feature_dict[c] for c in FEATURE_COLUMNS])
    return (x - SCALER.mean_) / SCALER.scale_


def main():
    print(f"scaler.mean_  = {dict(zip(FEATURE_COLUMNS, SCALER.mean_.round(4)))}")
    print(f"scaler.scale_ = {dict(zip(FEATURE_COLUMNS, SCALER.scale_.round(4)))}")
    print()

    rows = []
    for world_name, pairs in HAND_PICKED_PAIRS.items():
        sdf_path = erf.WORLDS_SDF_DIR / f"{world_name}.sdf"
        objects, cameras = erf.parse_world_sdf(sdf_path)
        camera = cameras[1]
        render = ray_cast_scene(camera, objects)
        sdf_source = erf.SDFGeometryRelationSource({world_name: objects})

        for name_a, name_b in pairs:
            if name_a not in objects or name_b not in objects:
                continue
            mask_a = render.owner == name_a
            mask_b = render.owner == name_b
            if not np.any(mask_a) or not np.any(mask_b):
                print(f"[{world_name}] {name_a}/{name_b}: not visible in this render, skipping")
                continue

            live = erf.LiveDepthFeatureSource(camera=camera)
            feats = live.compute(render.depth, mask_a, mask_b)

            x = np.array([[feats[c] for c in FEATURE_COLUMNS]])
            if np.isnan(x).any():
                print(f"[{world_name}] {name_a}/{name_b}: NaN feature, skipping "
                      f"({[c for c in FEATURE_COLUMNS if np.isnan(feats[c])]})")
                continue

            x_scaled = SCALER.transform(x)
            pred = int(MODEL.predict(x_scaled)[0])
            proba = MODEL.predict_proba(x_scaled)[0]
            z = zscore_row(feats)

            obj_a, obj_b = objects[name_a], objects[name_b]
            sdf_relation = sdf_source._derive(erf.world_aabb(obj_a), erf.world_aabb(obj_b))

            rows.append({
                "world": world_name, "pair": f"{name_a}/{name_b}",
                "sdf_relation": sdf_relation,
                "model_pred": "Contact" if pred == 1 else "Separate",
                "p_contact": round(float(proba[1]), 6),
                **{f"z_{c}": round(float(zv), 2) for c, zv in zip(FEATURE_COLUMNS, z)},
                **{c: round(float(feats[c]), 5) for c in FEATURE_COLUMNS},
            })

    print(f"{'world':<22s} {'pair':<28s} {'sdf_gt':<9s} {'pred':<9s} {'p(Contact)':<11s} "
          + " ".join(f"z_{c[:10]:<10s}" for c in FEATURE_COLUMNS))
    for r in rows:
        print(f"{r['world']:<22s} {r['pair']:<28s} {r['sdf_relation']:<9s} {r['model_pred']:<9s} "
              f"{r['p_contact']:<11.6f} "
              + " ".join(f"{r['z_'+c]:<12.2f}" for c in FEATURE_COLUMNS))

    print()
    print("=== Raw (unscaled) feature values ===")
    for r in rows:
        vals = ", ".join(f"{c}={r[c]:.4f}" for c in FEATURE_COLUMNS)
        print(f"{r['world']} {r['pair']} (sdf_gt={r['sdf_relation']}, pred={r['model_pred']}, "
              f"p={r['p_contact']:.4f}): {vals}")

    n_contact_pred = sum(1 for r in rows if r["model_pred"] == "Contact")
    print()
    print(f"Total pairs evaluated: {len(rows)}; predicted Contact: {n_contact_pred}; "
          f"predicted Separate: {len(rows) - n_contact_pred}")


if __name__ == "__main__":
    main()
