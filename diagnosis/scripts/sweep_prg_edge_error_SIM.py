#!/usr/bin/env python3
"""
diagnosis/scripts/sweep_prg_edge_error_SIM.py

T3: wider sweep of compute_prg_edge_error() outputs across all 6 staged
worlds, full adjacent-pair sweep (not just the 15 hand-picked pairs),
using the same ray-cast depth source as T1/T2 (no real hardware, no
DA-V2 -- see diagnosis/BINARY_MODEL_DIAGNOSIS.md H1 blocker).

Ground truth relation per pair: SDFGeometryRelationSource (static-pose
AABB heuristic) -- the only ground truth available without a running
physics sim / gazebo_contacts file in this environment. This is the
SAME style of heuristic that diagnosis/BINARY_MODEL_DIAGNOSIS.md's H2
shows disagrees with physics-verified gazebo_contact 60/101 times on
the one recoverable labeled dataset -- treat this sweep's "ground truth"
with that caveat, it is not physics-verified.

Does not modify any production file, model, or scaler. All numbers are
SIM / SYNTHETIC -- never citable as C1/C1-ext/C4/C4-ext evidence.
"""
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))

import numpy as np

import extract_relation_features as erf
from validate_live_depth_feature_source import ray_cast_scene
from classify_relation import classify_relation
from compute_prg_edge_error import compute_prg_edge_error

OUT_CSV = PROJECT_ROOT / "diagnosis" / "prg_edge_error_sweep_SIM.csv"
OUT_MD = PROJECT_ROOT / "diagnosis" / "PRG_EDGE_ERROR_SWEEP_SIM.md"

WORLDS = [
    "world1_baseline", "world2_dense_clutter", "world3_thin_objects",
    "world4_support_scene", "world5_occlusion_scene", "world6_dense_mixed",
]


def main():
    rows = []
    for world_name in WORLDS:
        sdf_path = erf.WORLDS_SDF_DIR / f"{world_name}.sdf"
        objects, cameras = erf.parse_world_sdf(sdf_path)
        camera = cameras[1]
        render = ray_cast_scene(camera, objects)
        sdf_source = erf.SDFGeometryRelationSource({world_name: objects})

        masks = {name: (render.owner == name) for name in objects if np.any(render.owner == name)}
        pairs = erf.enumerate_adjacent_pairs(masks)

        for name_a, name_b in pairs:
            mask_a, mask_b = masks[name_a], masks[name_b]
            live = erf.LiveDepthFeatureSource(camera=camera)
            feats = live.compute(render.depth, mask_a, mask_b)
            if any(isinstance(v, float) and np.isnan(v) for v in feats.values()):
                continue

            relation, confidence = classify_relation(feats)
            edge_error = compute_prg_edge_error(
                rgb=None, depth=render.depth, target_bbox=None,
                relation=relation, confidence=confidence,
                mask_a=mask_a, mask_b=mask_b,
            )

            obj_a, obj_b = objects[name_a], objects[name_b]
            gt_relation = sdf_source._derive(erf.world_aabb(obj_a), erf.world_aabb(obj_b))
            gt_touching = gt_relation != "Separate"
            pred_touching = relation != "Separate"
            wrong = gt_touching != pred_touching
            wrong_3way = gt_relation != relation

            rows.append({
                "world": world_name, "object_a": name_a, "object_b": name_b,
                "gt_relation": gt_relation, "pred_relation": relation,
                "pred_confidence": round(confidence, 6),
                "prg_edge_error": round(edge_error, 6),
                "binary_wrong": int(wrong), "three_way_wrong": int(wrong_3way),
            })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    errors = np.array([r["prg_edge_error"] for r in rows])
    wrong = np.array([r["binary_wrong"] for r in rows])
    wrong_3way = np.array([r["three_way_wrong"] for r in rows])

    def auroc(scores, labels):
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        greater = np.sum(pos[:, None] > neg[None, :])
        ties = np.sum(pos[:, None] == neg[None, :])
        return float((greater + 0.5 * ties) / (len(pos) * len(neg)))

    def point_biserial(scores, labels):
        if len(set(labels.tolist())) < 2 or np.std(scores) == 0:
            return float("nan")
        return float(np.corrcoef(scores, labels)[0, 1])

    auroc_binary = auroc(errors, wrong)
    pb_binary = point_biserial(errors, wrong)
    auroc_3way = auroc(errors, wrong_3way)
    pb_3way = point_biserial(errors, wrong_3way)

    by_relation = {}
    for r in rows:
        by_relation.setdefault(r["gt_relation"], []).append(r["prg_edge_error"])

    lines = []
    lines.append("# PRG Edge Error Sweep (SIM)")
    lines.append("")
    lines.append("SYNTHETIC / SIM ONLY -- ray-cast depth on staged SDF scenes, no")
    lines.append("real hardware, no DA-V2 (blocked, see BINARY_MODEL_DIAGNOSIS.md H1).")
    lines.append("Ground truth is SDFGeometryRelationSource (static-pose heuristic),")
    lines.append("NOT physics-verified gazebo_contact -- see caveat in this script's")
    lines.append("docstring and BINARY_MODEL_DIAGNOSIS.md H2. Not citable as C1/C1-ext/")
    lines.append("C4/C4-ext evidence. Reproduce with:")
    lines.append("`python3 diagnosis/scripts/sweep_prg_edge_error_SIM.py`")
    lines.append("")
    lines.append(f"Total adjacent pairs swept across all 6 worlds: {len(rows)}")
    lines.append("")
    lines.append("## prg_edge_error distribution per ground-truth relation")
    lines.append("")
    lines.append("| gt_relation | n | mean | std | min | max |")
    lines.append("|---|---|---|---|---|---|")
    for rel, vals in sorted(by_relation.items()):
        vals = np.array(vals)
        lines.append(f"| {rel} | {len(vals)} | {vals.mean():.4f} | {vals.std():.4f} | {vals.min():.4f} | {vals.max():.4f} |")
    lines.append("")
    lines.append("## Does high prg_edge_error co-occur with a wrong relation prediction?")
    lines.append("")
    lines.append(f"- Binary (touching vs. Separate) wrong-prediction rate: {wrong.mean():.3f} ({wrong.sum()}/{len(wrong)})")
    lines.append(f"- AUROC(prg_edge_error, binary_wrong) = {auroc_binary:.4f}")
    lines.append(f"- Point-biserial correlation(prg_edge_error, binary_wrong) = {pb_binary:.4f}")
    lines.append(f"- 3-way (Contact/Support/Separate) wrong-prediction rate: {wrong_3way.mean():.3f} ({wrong_3way.sum()}/{len(wrong_3way)})")
    lines.append(f"- AUROC(prg_edge_error, three_way_wrong) = {auroc_3way:.4f}")
    lines.append(f"- Point-biserial correlation(prg_edge_error, three_way_wrong) = {pb_3way:.4f}")
    lines.append("")
    lines.append("## prg_edge_error value range check")
    lines.append("")
    lines.append(f"- min={errors.min():.4f}, max={errors.max():.4f}, all in [0,1]: {bool(np.all((errors >= 0) & (errors <= 1)))}")
    lines.append(f"- constant? unique values: {len(set(np.round(errors, 6).tolist()))}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Given BINARY_MODEL_DIAGNOSIS.md's H3 finding that the frozen binary "
        "classifier is currently near-degenerate (predicts 'Separate' on "
        "almost every pair regardless of ground truth), any AUROC signal "
        "above 0.5 here is expected to come largely from "
        "compute_prg_edge_error()'s geometric_term (a raw depth-gap check "
        "against 'Separate' predictions), essentially independent of the "
        "classifier's own confidence -- NOT evidence that the two-signal "
        "(classifier confidence + geometric consistency) design is working "
        "as intended. This should be re-run once the classifier itself is "
        "fixed (see BINARY_MODEL_DIAGNOSIS.md recommendation) before drawing "
        "any conclusion about prg_edge_error's real diagnostic value."
    )

    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
