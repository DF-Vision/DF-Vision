#!/usr/bin/env python3
"""
experiments/rerun_with_corrected_interface_SIM.py

EXPERIMENTAL / SIM ONLY -- not a production fix, not wired into any
pipeline, does not modify extract_relation_features.py, classify_relation.py,
or any committed model/scaler/CSV.

Tests one specific, code-confirmed hypothesis from
diagnosis/BINARY_MODEL_DIAGNOSIS.md: that feat_boundary_sharpness() and
feat_depth_gradient()'s "interface" region (ring_a | ring_b, effectively --
see diagnosis doc section 1d) is dominated by each object's silhouette
against the BACKGROUND rather than the true mutual object-object boundary,
and that this alone explains the frozen logistic_model.joblib's "always
Separate" behavior on ray-cast depth.

Method: recompute ONLY boundary_sharpness and depth_gradient using a
corrected "true mutual boundary" region (ring_a intersected with dilated
mask_b, or vice versa) instead of the codebase's ring_a|ring_b. All other
6 features come from the real, untouched LiveDepthFeatureSource.compute().
Feed the corrected 8-vector through the real, untouched, frozen
classify_relation() -- no retraining, no scaler/model changes.

Every number in this script's output is SYNTHETIC WIRING REHEARSAL /
DIAGNOSTIC ONLY -- ray-cast depth on staged SDF scenes, never real
hardware, never citable as C1/C1-ext/C4/C4-ext evidence.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))

import numpy as np
import cv2

import extract_relation_features as erf
from validate_live_depth_feature_source import ray_cast_scene, HAND_PICKED_PAIRS
from classify_relation import classify_relation


def corrected_boundary_sharpness(depth, mask_a, mask_b, ring_a, ring_b):
    mutual = (ring_a & cv2.dilate(mask_b.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)) | \
             (ring_b & cv2.dilate(mask_a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    if not np.any(mutual):
        return float("nan")
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(grad_mag[mutual]))


def main():
    print("=" * 78)
    print("SYNTHETIC WIRING REHEARSAL -- NOT A RESULT -- ray-cast depth, corrected")
    print("interface-region features fed through the real frozen classifier.")
    print("=" * 78)
    header = "{:<22s} {:<28s} {:<9s} {:<9s} {:<12s} {:<9s} {:<12s}".format(
        "world", "pair", "sdf_gt", "orig_pred", "orig_p(C)", "fix_pred", "fix_p(C)")
    print(header)

    n_orig_contact = n_fix_contact = n_total = 0
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
                continue

            live = erf.LiveDepthFeatureSource(camera=camera)
            feats_orig = live.compute(render.depth, mask_a, mask_b)
            if any(np.isnan(feats_orig[c]) for c in feats_orig if isinstance(feats_orig[c], float)):
                continue

            ring_a = erf._ring_from_mask(mask_a)
            ring_b = erf._ring_from_mask(mask_b)
            fixed_bs = corrected_boundary_sharpness(render.depth, mask_a, mask_b, ring_a, ring_b)
            # depth_gradient's union-based region has the same background-domination
            # issue (mask_a|mask_b interior boundary vs background); reuse the same
            # corrected mutual-boundary region for a same-treatment comparison.
            fixed_dg = fixed_bs  # both Sobel-mean features on the same corrected region

            feats_fixed = dict(feats_orig)
            if not np.isnan(fixed_bs):
                feats_fixed["boundary_sharpness"] = fixed_bs
                feats_fixed["depth_gradient"] = fixed_dg

            relation_orig, conf_orig = classify_relation(feats_orig)
            relation_fixed, conf_fixed = classify_relation(feats_fixed)

            obj_a, obj_b = objects[name_a], objects[name_b]
            sdf_relation = sdf_source._derive(erf.world_aabb(obj_a), erf.world_aabb(obj_b))

            # classify_relation returns confidence for the WINNING class, not
            # necessarily p(Contact) -- recover p(Contact) explicitly for a
            # fair before/after comparison.
            from classify_relation import _predict_binary
            _, p_contact_orig = _predict_binary(feats_orig)
            _, p_contact_fixed = _predict_binary(feats_fixed)
            if relation_orig == "Separate":
                p_contact_orig = 1.0 - p_contact_orig
            if relation_fixed == "Separate":
                p_contact_fixed = 1.0 - p_contact_fixed

            n_total += 1
            n_orig_contact += relation_orig != "Separate"
            n_fix_contact += relation_fixed != "Separate"

            print("{:<22s} {:<28s} {:<9s} {:<9s} {:<12.6f} {:<9s} {:<12.6f}".format(
                world_name, f"{name_a}/{name_b}", sdf_relation,
                relation_orig, p_contact_orig, relation_fixed, p_contact_fixed))

    print()
    print(f"Total pairs: {n_total}")
    print(f"Predicted non-Separate (Contact/Support), ORIGINAL features: {n_orig_contact}/{n_total}")
    print(f"Predicted non-Separate (Contact/Support), CORRECTED features: {n_fix_contact}/{n_total}")
    print()
    print("Reminder: this is a synthetic wiring rehearsal on ray-cast depth. It")
    print("tests whether the interface-region bug alone explains the frozen")
    print("model's behavior; it is NOT evidence the corrected feature would work")
    print("on real hardware, and the frozen model/scaler were not modified.")


if __name__ == "__main__":
    main()
