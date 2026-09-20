#!/usr/bin/env python3
"""
explore/scripts/e3_evaluate_candidates.py

E3: evaluate geometric prg_edge_error candidates C1/C2/C3 against
explore/CRITERIA.md's pre-committed gates (a)-(f), on the 5 non-held-out
worlds first, select one candidate by the pre-committed rule, then
evaluate the selected candidate on world6_dense_mixed exactly once.

Uses object_a's bounding box (from the ray-cast render's own mask, per
the arbitrary-but-fixed "object_a is the target" convention stated in
CRITERIA.md/this script) as target_bbox for every candidate call.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "explore" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "retrain" / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import extract_relation_features as erf
from validate_live_depth_feature_source import ray_cast_scene
from noise_model import apply_depth_noise
import geometric_prg_error as gpe

RANDOM_STATE = 20260920
DATA_CSV = PROJECT_ROOT / "retrain" / "data" / "relation_dataset_v2_SIM.csv"
ORACLE_CSV = PROJECT_ROOT / "explore" / "data" / "oracle_features_SIM.csv"
HELD_OUT_WORLD = "world6_dense_mixed"
NOISE_LEVELS = [0.0, 1.0, 2.0, 3.0, 5.0]
BLEND_LEVELS_PX = [0, 1, 2, 4, 8]
WRONG_GAP_THRESHOLD_M = 0.01  # 1cm, per criterion (c)

WORLDS_TRAIN = ["world1_baseline", "world2_dense_clutter", "world3_thin_objects",
                "world4_support_scene", "world5_occlusion_scene"]


def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1)


def build_scene_cache(worlds):
    """One ray-cast render per (world, viewpoint) actually used in the
    dataset, at noise_level=0 (clean) -- the noise/blend sweeps in (b)
    apply their own controlled noise on top of this clean base, so it is
    computed once and reused."""
    df = pd.read_csv(DATA_CSV)
    df = df[df["world_id"].isin(worlds)]
    cache = {}
    for world_name in worlds:
        objects, cameras = erf.parse_world_sdf(erf.WORLDS_SDF_DIR / f"{world_name}.sdf")
        for vp in df[df["world_id"] == world_name]["viewpoint"].unique():
            vp = int(vp)
            if vp not in cameras:
                continue
            camera = cameras[vp]
            render = ray_cast_scene(camera, objects)
            cache[(world_name, vp)] = (objects, camera, render)
    return cache, df


def criterion_a_b(cache, df):
    print("\n" + "=" * 78)
    print("(a) Non-constant, in [0,1], deterministic")
    print("=" * 78)
    all_scores = {name: [] for name in gpe.CANDIDATES}
    for _, row in df.iterrows():
        key = (row["world_id"], int(row["viewpoint"]))
        if key not in cache:
            continue
        objects, camera, render = cache[key]
        mask_a = render.owner == row["object_a"]
        if not np.any(mask_a):
            continue
        bbox = mask_to_bbox(mask_a)
        for name, fn in gpe.CANDIDATES.items():
            s1 = fn(None, render.depth, bbox)
            s2 = fn(None, render.depth, bbox)  # determinism check
            assert s1 == s2, f"{name} not deterministic: {s1} vs {s2}"
            all_scores[name].append(s1)

    results = {}
    for name, scores in all_scores.items():
        scores = np.array(scores)
        in_range = bool(np.all((scores >= 0) & (scores <= 1)))
        n_unique = len(set(np.round(scores, 6).tolist()))
        passed = in_range and n_unique >= 3
        print(f"{name}: n={len(scores)}, range=[{scores.min():.4f},{scores.max():.4f}], "
              f"unique={n_unique}, in_range={in_range}, deterministic=True -> "
              f"{'PASS' if passed else 'FAIL'}")
        results[name] = passed
    return results


def criterion_b_sensitivity(cache):
    print("\n" + "=" * 78)
    print("(b) Sensitivity: Spearman rho > 0.6 with noise level AND edge-blend level, "
          "per world, >= 4 of 5 worlds")
    print("=" * 78)
    results = {name: {} for name in gpe.CANDIDATES}

    for world_name in WORLDS_TRAIN:
        keys = [k for k in cache if k[0] == world_name]
        if not keys:
            continue
        objects, camera, render = cache[keys[0]]
        masks = {n: (render.owner == n) for n in objects if np.any(render.owner == n)}
        pairs = erf.enumerate_adjacent_pairs(masks)
        bboxes = [mask_to_bbox(masks[a]) for a, b in pairs]
        bboxes = [b for b in bboxes if b is not None]

        for name, fn in gpe.CANDIDATES.items():
            noise_levels_seen, noise_scores = [], []
            for level in NOISE_LEVELS:
                noisy = apply_depth_noise(render.depth, level=level, seed=RANDOM_STATE + int(level))
                for bbox in bboxes:
                    noise_scores.append(fn(None, noisy, bbox))
                    noise_levels_seen.append(level)
            rho_noise, _ = spearmanr(noise_levels_seen, noise_scores)

            blend_levels_seen, blend_scores = [], []
            for blend_px in BLEND_LEVELS_PX:
                blended = apply_depth_noise(render.depth, level=1.0 if blend_px > 0 else 0.0,
                                             seed=RANDOM_STATE, edge_blend_px=blend_px,
                                             sigma0=0.0, k=0.0)  # isolate blending only
                for bbox in bboxes:
                    blend_scores.append(fn(None, blended, bbox))
                    blend_levels_seen.append(blend_px)
            rho_blend, _ = spearmanr(blend_levels_seen, blend_scores)

            results[name][world_name] = (rho_noise, rho_blend)
            print(f"{name} / {world_name}: rho_noise={rho_noise:.3f}, rho_blend={rho_blend:.3f}")

    summary = {}
    for name in gpe.CANDIDATES:
        n_pass_noise = sum(1 for w, (rn, rb) in results[name].items() if rn > 0.6)
        n_pass_blend = sum(1 for w, (rn, rb) in results[name].items() if rb > 0.6)
        n_worlds = len(results[name])
        passed = n_pass_noise >= 4 and n_pass_blend >= 4
        print(f"{name}: noise rho>0.6 on {n_pass_noise}/{n_worlds} worlds, "
              f"blend rho>0.6 on {n_pass_blend}/{n_worlds} worlds -> {'PASS' if passed else 'FAIL'}")
        summary[name] = passed
    return summary, results


def auroc(scores, labels):
    scores, labels = np.array(scores), np.array(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = np.sum(pos[:, None] > neg[None, :])
    ties = np.sum(pos[:, None] == neg[None, :])
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def bootstrap_auroc_ci(scores, labels, worlds, n_boot=2000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    unique_worlds = np.unique(worlds)
    accs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_worlds, size=len(unique_worlds), replace=True)
        idx = np.concatenate([np.where(worlds == w)[0] for w in sampled])
        if len(set(np.array(labels)[idx].tolist())) < 2:
            continue
        accs.append(auroc(np.array(scores)[idx], np.array(labels)[idx]))
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def criterion_c_validity(cache, df):
    print("\n" + "=" * 78)
    print("(c) Validity: AUROC > 0.70, CI lower bound > 0.55 vs |estimated_gap - true_gap| > 1cm")
    print("=" * 78)
    oracle_df = pd.read_csv(ORACLE_CSV)
    oracle_lookup = {(r.world_id, r.object_a, r.object_b): r.true_min_gap for r in oracle_df.itertuples()}

    rows = []
    for _, row in df.iterrows():
        key = (row["world_id"], int(row["viewpoint"]))
        if key not in cache:
            continue
        objects, camera, render = cache[key]
        mask_a = render.owner == row["object_a"]
        if not np.any(mask_a):
            continue
        bbox = mask_to_bbox(mask_a)
        true_gap = oracle_lookup.get((row["world_id"], row["object_a"], row["object_b"]))
        if true_gap is None:
            continue
        c1_internal = gpe._c1_internals(render.depth, bbox)
        if c1_internal is None:
            continue
        estimated_gap, sigma_gap, c1_score = c1_internal
        c2_score = gpe.geometric_c2_boundary_sharpness(None, render.depth, bbox)
        c3_score = max(c1_score, c2_score)
        error = abs(estimated_gap - true_gap)
        rows.append({
            "world": row["world_id"], "wrong": int(error > WRONG_GAP_THRESHOLD_M),
            "C1": c1_score, "C2": c2_score, "C3": c3_score,
        })

    val_df = pd.DataFrame(rows)
    print(f"n={len(val_df)}, wrong fraction={val_df['wrong'].mean():.3f}")
    results = {}
    for name in gpe.CANDIDATES:
        auc = auroc(val_df[name], val_df["wrong"])
        lo, hi = bootstrap_auroc_ci(val_df[name], val_df["wrong"], val_df["world"].to_numpy())
        passed = auc > 0.70 and lo > 0.55
        print(f"{name}: AUROC={auc:.4f}, CI=[{lo:.4f},{hi:.4f}] -> {'PASS' if passed else 'FAIL'}")
        results[name] = {"auroc": auc, "ci": (lo, hi), "pass": passed}
    return results, val_df


def criterion_d_baselines(cache, df, val_df):
    print("\n" + "=" * 78)
    print("(d) Beats dumb baselines (constant, single-feature) by a margin whose CI excludes 0")
    print("=" * 78)
    # Constant baseline.
    const_scores = np.full(len(val_df), 0.5)
    const_auc = auroc(const_scores, val_df["wrong"])
    print(f"constant(0.5) baseline: AUROC={const_auc:.4f} (undefined/0.5 expected -- a constant "
          f"score has no ranking power)")

    # Single-feature baseline: raw boundary_sharpness, min-max scaled on non-held-out data.
    raw_bs = []
    for _, row in df.iterrows():
        key = (row["world_id"], int(row["viewpoint"]))
        if key not in cache:
            continue
        _, _, render = cache[key]
        mask_a = render.owner == row["object_a"]
        if not np.any(mask_a):
            continue
        bbox = mask_to_bbox(mask_a)
        bs = gpe._raw_boundary_sharpness(render.depth, bbox)
        if bs is not None:
            raw_bs.append(bs)
    raw_bs = np.array(raw_bs)
    lo_bs, hi_bs = np.percentile(raw_bs, [5, 95])
    print(f"single-feature baseline fit: boundary_sharpness 5th/95th percentile = [{lo_bs:.4f},{hi_bs:.4f}] "
          f"(fit on non-held-out worlds only)")

    rows = []
    oracle_df = pd.read_csv(ORACLE_CSV)
    oracle_lookup = {(r.world_id, r.object_a, r.object_b): r.true_min_gap for r in oracle_df.itertuples()}
    for _, row in df.iterrows():
        key = (row["world_id"], int(row["viewpoint"]))
        if key not in cache:
            continue
        _, _, render = cache[key]
        mask_a = render.owner == row["object_a"]
        if not np.any(mask_a):
            continue
        bbox = mask_to_bbox(mask_a)
        true_gap = oracle_lookup.get((row["world_id"], row["object_a"], row["object_b"]))
        if true_gap is None:
            continue
        c1_internal = gpe._c1_internals(render.depth, bbox)
        if c1_internal is None:
            continue
        estimated_gap, _, _ = c1_internal
        error = abs(estimated_gap - true_gap)
        bs = gpe._raw_boundary_sharpness(render.depth, bbox)
        sf_score = 1.0 - np.clip((bs - lo_bs) / max(hi_bs - lo_bs, 1e-9), 0, 1) if bs is not None else 0.5
        rows.append({"world": row["world_id"], "wrong": int(error > WRONG_GAP_THRESHOLD_M), "sf": sf_score})
    sf_df = pd.DataFrame(rows)
    sf_auc = auroc(sf_df["sf"], sf_df["wrong"])
    print(f"single-feature(boundary_sharpness) baseline: AUROC={sf_auc:.4f}")

    assert len(sf_df) == len(val_df), (
        f"single-feature baseline row count ({len(sf_df)}) must match candidate "
        f"row count ({len(val_df)}) -- both loops apply identical filters over "
        f"the same df, a mismatch means a filter diverged and the paired "
        f"bootstrap below would silently misalign rows."
    )
    sf_scores_aligned = sf_df["sf"].to_numpy()
    wrong_aligned = val_df["wrong"].to_numpy()

    results = {}
    for name in gpe.CANDIDATES:
        cand_scores = val_df[name].to_numpy()
        worlds = val_df["world"].to_numpy()
        rng = np.random.default_rng(RANDOM_STATE)
        unique_worlds = np.unique(worlds)
        diffs_const, diffs_sf = [], []
        for _ in range(2000):
            sampled = rng.choice(unique_worlds, size=len(unique_worlds), replace=True)
            idx = np.concatenate([np.where(worlds == w)[0] for w in sampled])
            if len(set(wrong_aligned[idx].tolist())) < 2:
                continue
            a1 = auroc(cand_scores[idx], wrong_aligned[idx])
            a2 = auroc(np.full(len(idx), 0.5), wrong_aligned[idx])
            a3 = auroc(sf_scores_aligned[idx], wrong_aligned[idx])
            diffs_const.append(a1 - a2)
            diffs_sf.append(a1 - a3)
        lo_c, hi_c = np.percentile(diffs_const, [2.5, 97.5])
        lo_s, hi_s = np.percentile(diffs_sf, [2.5, 97.5])
        beats_const = lo_c > 0
        beats_sf = lo_s > 0
        print(f"{name}: margin vs constant CI=[{lo_c:.4f},{hi_c:.4f}] ({'beats' if beats_const else 'no'}), "
              f"margin vs single-feature CI=[{lo_s:.4f},{hi_s:.4f}] ({'beats' if beats_sf else 'no'})")
        results[name] = beats_const and beats_sf
    return results


def criterion_e_collinearity(cache, df):
    print("\n" + "=" * 78)
    print("(e) Collinearity with true-occlusion proxy and object-camera distance (reporting only)")
    print("=" * 78)
    rows = []
    for _, row in df.iterrows():
        key = (row["world_id"], int(row["viewpoint"]))
        if key not in cache:
            continue
        objects, camera, render = cache[key]
        mask_a = render.owner == row["object_a"]
        if not np.any(mask_a):
            continue
        bbox = mask_to_bbox(mask_a)
        distance = float(np.median(render.depth[mask_a]))
        rows.append({"C1": gpe.geometric_c1_gap_uncertainty(None, render.depth, bbox),
                     "C2": gpe.geometric_c2_boundary_sharpness(None, render.depth, bbox),
                     "distance": distance, "mask_px": int(mask_a.sum())})
    e_df = pd.DataFrame(rows)
    e_df["C3"] = e_df[["C1", "C2"]].max(axis=1)
    # Use mask pixel COUNT (inverse relates to occlusion/distance) as the
    # occlusion proxy instead of the placeholder above -- smaller visible
    # area at a given object size plausibly indicates more occlusion or
    # more distance; reported honestly as a proxy, not a validated measure.
    for name in gpe.CANDIDATES:
        r_dist_pearson = e_df[name].corr(e_df["distance"], method="pearson")
        r_dist_spearman = e_df[name].corr(e_df["distance"], method="spearman")
        r_occ_pearson = e_df[name].corr(e_df["mask_px"], method="pearson")
        r_occ_spearman = e_df[name].corr(e_df["mask_px"], method="spearman")
        flag = any(abs(v) > 0.7 for v in [r_dist_pearson, r_dist_spearman, r_occ_pearson, r_occ_spearman])
        print(f"{name}: corr(distance) pearson={r_dist_pearson:.3f} spearman={r_dist_spearman:.3f}; "
              f"corr(mask_px, occlusion proxy) pearson={r_occ_pearson:.3f} spearman={r_occ_spearman:.3f} "
              f"{'[FLAG >0.7]' if flag else ''}")
    return e_df


def main():
    print("Building scene cache for 5 non-held-out worlds...")
    cache, df = build_scene_cache(WORLDS_TRAIN)

    a_results = criterion_a_b(cache, df)  # (a) determinism/range
    b_results, b_detail = criterion_b_sensitivity(cache)
    c_results, val_df = criterion_c_validity(cache, df)
    d_results = criterion_d_baselines(cache, df, val_df)
    e_df = criterion_e_collinearity(cache, df)

    print("\n" + "=" * 78)
    print("SELECTION (per the pre-committed rule in explore/CRITERIA.md)")
    print("=" * 78)
    eligible = [name for name in gpe.CANDIDATES if a_results[name] and b_results[name]]
    print(f"Candidates passing (a) and (b): {eligible}")
    if not eligible:
        print("NONE of C1/C2/C3 pass both (a) and (b) -- per the pre-committed rule, "
              "no candidate is selected. This is reported as a negative result.")
        return

    best = max(eligible, key=lambda n: c_results[n]["auroc"])
    print(f"Selected by highest (c) AUROC among eligible candidates: {best} "
          f"(AUROC={c_results[best]['auroc']:.4f})")
    print(f"\n(c) PASS: {c_results[best]['pass']}, (d) PASS: {d_results[best]}")

    if not (c_results[best]["pass"] and d_results[best]):
        print(f"\n{best} does not clear (c) and/or (d) either -- reported as a negative "
              f"result for ALL of C1/C2/C3, not silently promoted.")
        return

    print(f"\n{best} clears (a),(b),(c),(d) on non-held-out worlds. "
          f"Proceeding to evaluate on world6_dense_mixed exactly once.")


if __name__ == "__main__":
    main()
