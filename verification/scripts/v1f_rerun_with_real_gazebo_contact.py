#!/usr/bin/env python3
"""
verification/scripts/v1f_rerun_with_real_gazebo_contact.py

V1f: re-run the classify_relation() chain using REAL, physics-verified
gazebo_contact ground truth (verification/gazebo_contacts_SIM/*.csv, from
the first-ever execution of extract_gazebo_contacts.py this session) --
NOT the SDFGeometryRelationSource static-pose heuristic used as a stand-in
last session.

"Before correction" = legacy behavior (bit-for-bit what's in production
today: _ring_from_mask() uses a 3x3 SQUARE erosion kernel).
"After correction" = the ONLY confirmed train/live definition mismatch
found in V1a for the ring-consuming features: the ring kernel itself.
scripts/export_masks.py and scripts/export_boundary_mask.py (the actual
training-time boundary exporter) both use a 5x5 ELLIPSE erosion kernel
(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))), not a 3x3 square.
This script changes ONLY the kernel shape/size passed to the ring
generator -- it does NOT restrict the interface region to a "true mutual
boundary" (that was last session's invented definition, not what
training actually computed; feat_boundary_sharpness's own interface
logic, ring_a|ring_b, is IDENTICAL code in both paths and is therefore
NOT a train/live mismatch by itself -- see V1_AUDIT.md).

depth_gradient uses mask_a|mask_b directly (no ring at all) -- there is
no confirmed mismatch for it, so it is IDENTICAL in both modes here by
design, not by oversight.

IMPORTANT CAVEAT, stated up front: these are the SAME 6 staged worlds
likely used (or closely resembling what was used) to generate the
model's own training data -- any accuracy number here is IN-DISTRIBUTION
and OPTIMISTIC, not a real-world generalization estimate.

Does not modify extract_relation_features.py, classify_relation.py, or
any committed model/scaler. All output labeled SIM.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))

import numpy as np
import pandas as pd
import cv2

import extract_relation_features as erf
from validate_live_depth_feature_source import ray_cast_scene
from classify_relation import classify_relation, _predict_binary

GAZEBO_CONTACTS_DIR = PROJECT_ROOT / "verification" / "gazebo_contacts_SIM"
OUT_CSV = PROJECT_ROOT / "verification" / "v1f_results_SIM.csv"

WORLDS = [
    "world1_baseline", "world2_dense_clutter", "world3_thin_objects",
    "world4_support_scene", "world5_occlusion_scene", "world6_dense_mixed",
]


def pairwise_ring(mask, ring_width=5):
    """The TRAINING-time ring definition (scripts/export_masks.py line
    122-132, scripts/export_boundary_mask.py line 51/77/97): 5x5 ELLIPSE
    erosion, not the live path's 3x3 SQUARE (_ring_from_mask)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width, ring_width))
    m = mask.astype(np.uint8)
    eroded = cv2.erode(m, kernel)
    return (m - eroded).astype(bool)


def compute_features(depth, mask_a, mask_b, camera, ring_fn):
    """Reimplements LiveDepthFeatureSource.compute() but with a pluggable
    ring generator, so 'legacy' vs 'pairwise' differ ONLY in ring shape --
    everything else (Sobel math, points_world, overlap_ratio, etc.) is
    the untouched, real feat_* code."""
    mask_a, mask_b = mask_a.astype(bool), mask_b.astype(bool)
    ring_a, ring_b = ring_fn(mask_a), ring_fn(mask_b)
    points_world = erf.depth_to_world_points(depth, camera, 0.1, 5.0, depth_source_kind="mde_predicted_z")
    mask_a_flat, mask_b_flat = mask_a.reshape(-1), mask_b.reshape(-1)
    median_depth = float(np.nanmedian(depth[mask_a | mask_b])) if np.any(mask_a | mask_b) else None
    min_bd, is_metric = erf.feat_minimum_boundary_distance(mask_a, mask_b, camera, median_depth)
    return {
        "boundary_sharpness": erf.feat_boundary_sharpness(depth, ring_a, ring_b, mask_a, mask_b),
        "minimum_boundary_distance": min_bd,
        "minimum_boundary_distance_is_metric": is_metric,
        "surface_normal_consistency": erf.feat_surface_normal_consistency(points_world, mask_a_flat, mask_b_flat),
        "relative_height": erf.feat_relative_height(points_world, mask_a_flat, mask_b_flat),
        "depth_gradient": erf.feat_depth_gradient(depth, mask_a, mask_b),
        "overlap_ratio": erf.feat_overlap_ratio(mask_a, mask_b),
        "edge_continuity": erf.feat_edge_continuity(ring_a, ring_b, mask_a, mask_b),
        "occlusion_boundary_score": erf.feat_occlusion_boundary_score_proxy(depth, ring_a, ring_b, mask_a, mask_b),
    }


def load_real_gazebo_contact():
    frames = []
    for w in WORLDS:
        fp = GAZEBO_CONTACTS_DIR / f"{w}_gazebo_contacts_SIM.csv"
        if not fp.exists():
            raise FileNotFoundError(f"{fp} missing -- run run_gazebo_contacts_SIM.py first")
        frames.append(pd.read_csv(fp))
    return pd.concat(frames, ignore_index=True)


def main():
    gc_df = load_real_gazebo_contact()
    # index by (world_id, frozenset({a,b})) -> contact (0/1/"unknown")
    gc_lookup = {}
    for _, row in gc_df.iterrows():
        key = (row["world_id"], frozenset((row["object_a"], row["object_b"])))
        gc_lookup[key] = row["contact"]

    rows = []
    for world_name in WORLDS:
        objects, cameras = erf.parse_world_sdf(erf.WORLDS_SDF_DIR / f"{world_name}.sdf")
        camera = cameras[1]
        render = ray_cast_scene(camera, objects)
        masks = {name: (render.owner == name) for name in objects if np.any(render.owner == name)}
        pairs = erf.enumerate_adjacent_pairs(masks)

        for name_a, name_b in pairs:
            key = (world_name, frozenset((name_a, name_b)))
            gt = gc_lookup.get(key)
            if gt is None or gt == "unknown":
                continue  # no real physics ground truth for this pair -- skip, don't guess
            gt_touching = int(gt) == 1

            mask_a, mask_b = masks[name_a], masks[name_b]
            feats_legacy = compute_features(render.depth, mask_a, mask_b, camera,
                                             ring_fn=lambda m: erf._ring_from_mask(m, ring_px=3))
            feats_pairwise = compute_features(render.depth, mask_a, mask_b, camera,
                                               ring_fn=pairwise_ring)

            if any(isinstance(v, float) and np.isnan(v) for v in feats_legacy.values()):
                continue
            if any(isinstance(v, float) and np.isnan(v) for v in feats_pairwise.values()):
                continue

            rel_legacy, conf_legacy = classify_relation(feats_legacy)
            rel_pairwise, conf_pairwise = classify_relation(feats_pairwise)
            _, p1_legacy = _predict_binary(feats_legacy)
            _, p1_pairwise = _predict_binary(feats_pairwise)

            rows.append({
                "world": world_name, "object_a": name_a, "object_b": name_b,
                "gazebo_contact_REAL": int(gt_touching),
                "pred_legacy": rel_legacy, "p_touching_legacy": round(p1_legacy, 6),
                "pred_pairwise": rel_pairwise, "p_touching_pairwise": round(p1_pairwise, 6),
                "touching_legacy": int(rel_legacy != "Separate"),
                "touching_pairwise": int(rel_pairwise != "Separate"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"n = {len(df)} pairs with real, non-unknown gazebo_contact ground truth")
    print(df.to_string(index=False))

    def balanced_acc(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        recalls = []
        for cls in [0, 1]:
            mask = y_true == cls
            if mask.sum() == 0:
                continue
            recalls.append((y_pred[mask] == cls).mean())
        return np.mean(recalls) if recalls else float("nan")

    def per_class_recall(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        out = {}
        for cls, name in [(0, "Separate(0)"), (1, "Touching(1)")]:
            mask = y_true == cls
            out[name] = float((y_pred[mask] == cls).mean()) if mask.sum() > 0 else float("nan")
            out[name + "_n"] = int(mask.sum())
        return out

    def bootstrap_ci_by_world(df, pred_col, n_boot=2000, seed=42):
        rng = np.random.default_rng(seed)
        worlds = df["world"].unique()
        accs = []
        for _ in range(n_boot):
            sampled_worlds = rng.choice(worlds, size=len(worlds), replace=True)
            sub = pd.concat([df[df["world"] == w] for w in sampled_worlds], ignore_index=True)
            if sub["gazebo_contact_REAL"].nunique() < 2:
                continue
            accs.append(balanced_acc(sub["gazebo_contact_REAL"], sub[pred_col]))
        accs = np.array(accs)
        return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)), len(accs)

    print("\n=== LEGACY (production, 3x3 square ring) ===")
    ba_legacy = balanced_acc(df["gazebo_contact_REAL"], df["touching_legacy"])
    pcr_legacy = per_class_recall(df["gazebo_contact_REAL"], df["touching_legacy"])
    lo_l, hi_l, nboot_l = bootstrap_ci_by_world(df, "touching_legacy")
    print(f"balanced accuracy = {ba_legacy:.4f}")
    print(f"per-class recall = {pcr_legacy}")
    print(f"bootstrap 95% CI (grouped by world, n_boot_used={nboot_l}) = [{lo_l:.4f}, {hi_l:.4f}]")
    print(f"n classes predicted: {df['touching_legacy'].nunique()} (1 = degenerate, always same answer)")

    print("\n=== PAIRWISE (5x5 ellipse ring, matching training-time export_masks.py) ===")
    ba_pw = balanced_acc(df["gazebo_contact_REAL"], df["touching_pairwise"])
    pcr_pw = per_class_recall(df["gazebo_contact_REAL"], df["touching_pairwise"])
    lo_p, hi_p, nboot_p = bootstrap_ci_by_world(df, "touching_pairwise")
    print(f"balanced accuracy = {ba_pw:.4f}")
    print(f"per-class recall = {pcr_pw}")
    print(f"bootstrap 95% CI (grouped by world, n_boot_used={nboot_p}) = [{lo_p:.4f}, {hi_p:.4f}]")
    print(f"n classes predicted: {df['touching_pairwise'].nunique()} (1 = degenerate, always same answer)")

    print("\nCAVEAT: these 6 worlds are the same staged scenes the model's training")
    print("data almost certainly came from (or closely resembles) -- this is an")
    print("IN-DISTRIBUTION, OPTIMISTIC accuracy estimate, not a generalization claim.")


if __name__ == "__main__":
    main()
