#!/usr/bin/env python3
"""
retrain/scripts/r2_generate_dataset.py

R2: regenerate labeled training data with provenance, fixing the lost-CSV
problem for good by committing the output with a manifest.

TRAIN/SERVE PARITY (rule 7): every feature vector here is produced by
calling the real, unmodified relation_classifier.scripts.extract_relation_features.LiveDepthFeatureSource.compute()
-- the exact same class production classify_relation.py/prg_client.py use
at inference. No separate training-time feature code exists anywhere in
this script.

Labels: gazebo_contact (physics-verified, from
verification/gazebo_contacts_SIM/*.csv, obtained last session by actually
running extract_gazebo_contacts.py for the first time) for
touching-vs-not. For touching pairs, Contact vs Support is derived from
SDF-authored geometry (vertical-stacking + xy-overlap test, same logic as
retrain/scripts/r1_cheap_checks.py's R1b check, itself derived from
SDFGeometryRelationSource's own Support test) -- gazebo's contact sensor
only reports binary touching, not the 3-way split.

Depth source: the ray-cast renderer (validate_live_depth_feature_source.ray_cast_scene,
validated 182/0/18 last-last session) plus retrain.scripts.noise_model's
noise model, at a RANDOMIZED level per (world, viewpoint) scene so the
model isn't tuned to one noise level. Every row logs its exact noise
level and seed for reproducibility.

Scope: 3 camera viewpoints per world (1, 2, 3) x all adjacent pairs
(enumerate_adjacent_pairs -- includes both touching and near-but-separate
pairs) x 1 randomized noise level per (world, viewpoint). All 6 worlds are
included in the generated file (including world6_dense_mixed, the
pre-designated held-out test world per GATES.md) -- separating world6 out
for the held-out test happens at ANALYSIS time (R3/R5), not generation
time; generating its features is not "peeking" at results.

Output: retrain/data/relation_dataset_v2_SIM.csv + a manifest with SHA256,
row counts, generator commit, seeds, date.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "retrain" / "scripts"))

import numpy as np
import pandas as pd

import extract_relation_features as erf
from validate_live_depth_feature_source import ray_cast_scene
from noise_model import apply_depth_noise

RANDOM_STATE = 20260920
NOISE_LEVELS = [0.0, 1.0, 2.0, 3.0, 5.0]  # matches GATES.md's "realistic" range
VIEWPOINTS = [1, 2, 3, 4, 5, 6]  # all 6 available per world (extended from
# an initial 3-viewpoint run, which produced only 155 rows -- generation
# is cheap (~4 min for 3 viewpoints x 6 worlds), so using all 6 to better
# hit the "several hundred" target).
SUPPORT_XY_EPS = 0.02
SUPPORT_Z_EPS = 0.02

GC_DIR = PROJECT_ROOT / "verification" / "gazebo_contacts_SIM"
OUT_DIR = PROJECT_ROOT / "retrain" / "data"
OUT_CSV = OUT_DIR / "relation_dataset_v2_SIM.csv"
MANIFEST = OUT_DIR / "MANIFEST.md"

WORLDS = [
    "world1_baseline", "world2_dense_clutter", "world3_thin_objects",
    "world4_support_scene", "world5_occlusion_scene", "world6_dense_mixed",
]

FEATURE_COLUMNS = [
    "boundary_sharpness", "minimum_boundary_distance", "surface_normal_consistency",
    "relative_height", "depth_gradient", "overlap_ratio", "edge_continuity",
    "occlusion_boundary_score",
]


def load_gazebo_contact_lookup():
    lookup = {}
    for w in WORLDS:
        fp = GC_DIR / f"{w}_gazebo_contacts_SIM.csv"
        df = pd.read_csv(fp)
        for _, r in df.iterrows():
            key = (w, frozenset((r["object_a"], r["object_b"])))
            lookup[key] = r["contact"]
    return lookup


def derive_relation(world_name, objects, name_a, name_b, gc_lookup):
    key = (world_name, frozenset((name_a, name_b)))
    gt = gc_lookup.get(key)
    if gt is None or gt == "unknown":
        return None
    if int(gt) == 0:
        return "Separate"
    obj_a, obj_b = objects[name_a], objects[name_b]
    aabb_a, aabb_b = erf.world_aabb(obj_a), erf.world_aabb(obj_b)
    (ax0, ay0, az0), (ax1, ay1, az1) = aabb_a
    (bx0, by0, bz0), (bx1, by1, bz1) = aabb_b
    xy_overlap = (erf._interval_gap(ax0, ax1, bx0, bx1) <= SUPPORT_XY_EPS
                  and erf._interval_gap(ay0, ay1, by0, by1) <= SUPPORT_XY_EPS)
    a_below_b = abs(az1 - bz0) <= SUPPORT_Z_EPS
    b_below_a = abs(bz1 - az0) <= SUPPORT_Z_EPS
    return "Support" if (xy_overlap and (a_below_b or b_below_a)) else "Contact"


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
    except Exception:
        return "UNKNOWN"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gc_lookup = load_gazebo_contact_lookup()
    generator_commit = get_git_commit()

    rng_master = np.random.default_rng(RANDOM_STATE)
    rows = []
    label_counts = {}

    for world_name in WORLDS:
        objects, cameras = erf.parse_world_sdf(erf.WORLDS_SDF_DIR / f"{world_name}.sdf")
        for vp in VIEWPOINTS:
            if vp not in cameras:
                print(f"[SKIP] {world_name} viewpoint {vp}: no camera")
                continue
            camera = cameras[vp]
            render = ray_cast_scene(camera, objects)
            noise_level = float(rng_master.choice(NOISE_LEVELS))
            scene_seed = int(rng_master.integers(0, 2**31 - 1))
            noisy_depth = apply_depth_noise(render.depth, level=noise_level, seed=scene_seed)

            masks = {name: (render.owner == name) for name in objects if np.any(render.owner == name)}
            pairs = erf.enumerate_adjacent_pairs(masks)
            live = erf.LiveDepthFeatureSource(camera=camera)

            for name_a, name_b in pairs:
                relation = derive_relation(world_name, objects, name_a, name_b, gc_lookup)
                if relation is None:
                    continue  # unresolved gazebo_contact -- don't guess
                mask_a, mask_b = masks[name_a], masks[name_b]
                feats = live.compute(noisy_depth, mask_a, mask_b)
                if any(isinstance(v, float) and np.isnan(v) for v in feats.values()):
                    continue

                row = {
                    "world_id": world_name, "viewpoint": vp,
                    "object_a": name_a, "object_b": name_b,
                    "relation": relation,
                    "noise_level": noise_level, "scene_seed": scene_seed,
                    "generator_commit": generator_commit,
                }
                for c in FEATURE_COLUMNS:
                    row[c] = feats[c]
                rows.append(row)
                label_counts[(world_name, relation)] = label_counts.get((world_name, relation), 0) + 1

            print(f"{world_name} vp={vp}: noise_level={noise_level}, "
                  f"{len(pairs)} pairs checked, running total rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    import hashlib
    sha256 = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()

    print(f"\nWrote {len(df)} rows -> {OUT_CSV}")
    print(f"SHA256: {sha256}")
    print("\nLabel counts per world:")
    counts_df = df.groupby(["world_id", "relation"]).size().unstack(fill_value=0)
    print(counts_df.to_string())
    print("\nOverall label counts:")
    print(df["relation"].value_counts().to_string())
    print("\nNoise level per (world, viewpoint):")
    print(df.groupby(["world_id", "viewpoint"])["noise_level"].first().to_string())

    manifest_lines = [
        "# retrain/data/relation_dataset_v2_SIM.csv -- Manifest",
        "",
        f"Generated: 2026-09-20 (autonomous retrain session)",
        f"Generator: retrain/scripts/r2_generate_dataset.py",
        f"Generator git commit (at generation time): {generator_commit}",
        f"Random seed: RANDOM_STATE={RANDOM_STATE}",
        f"SHA256 of {OUT_CSV.name}: {sha256}",
        f"Row count: {len(df)}",
        f"Viewpoints used per world: {VIEWPOINTS} (of 6 available)",
        f"Noise levels sampled from: {NOISE_LEVELS} (uniform random choice per (world,viewpoint), seeded)",
        "",
        "Label counts per world:",
        "```",
        counts_df.to_string(),
        "```",
        "",
        "Overall label counts:",
        "```",
        df["relation"].value_counts().to_string(),
        "```",
        "",
        "SIM / SYNTHETIC. Depth source: ray-cast renderer "
        "(relation_classifier/scripts/validate_live_depth_feature_source.ray_cast_scene) "
        "+ retrain/scripts/noise_model.py (documented, disclosed approximation of "
        "real sensor noise -- see that file's docstring). Labels: physics-verified "
        "gazebo_contact (touching vs not) + SDF-geometry-derived Support/Contact "
        "split for touching pairs. Not real hardware data. Not citable as "
        "C1/C1-ext/C4/C4-ext evidence.",
        "",
        "Features produced by the REAL, unmodified LiveDepthFeatureSource.compute() "
        "-- train/serve parity by construction (rule 7).",
        "",
        f"world6_dense_mixed rows ARE included in this file (generating them is not "
        f"'peeking' -- using them for model selection would be; see retrain/GATES.md "
        f"for the held-out-test discipline).",
    ]
    MANIFEST.write_text("\n".join(manifest_lines) + "\n")
    print(f"\nWrote manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
