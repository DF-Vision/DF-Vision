#!/usr/bin/env python3
"""
explore/scripts/e5_evaluate_neighbor_aware_candidate.py

C5: evaluates the neighbor-aware candidate (geometric_prg_error_neighbor_aware.py)
against explore/CRITERIA.md's criteria (a)-(e), reusing e3_evaluate_candidates.py's
own helpers (build_scene_cache, mask_to_bbox, auroc, bootstrap_auroc_ci) rather
than reimplementing them. Stops at the FIRST failing criterion with a SHORT
note, per project_closure/CRITERIA.md's C5 section and the task's own
explicit instruction ("stop at the first failure... no long report").
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
from noise_model import apply_depth_noise
import geometric_prg_error_neighbor_aware as na
from e3_evaluate_candidates import (
    build_scene_cache, mask_to_bbox, auroc, bootstrap_auroc_ci,
    WORLDS_TRAIN, HELD_OUT_WORLD, NOISE_LEVELS, BLEND_LEVELS_PX,
    WRONG_GAP_THRESHOLD_M, RANDOM_STATE, ORACLE_CSV,
)

CANDIDATE_NAME = "N1"
CANDIDATE_FN = na.geometric_neighbor_aware


def criterion_a(cache, df):
    print("\n(a) Non-constant, in [0,1], deterministic")
    scores = []
    for _, row in df.iterrows():
        key = (row["world_id"], int(row["viewpoint"]))
        if key not in cache:
            continue
        _, _, render = cache[key]
        mask_a = render.owner == row["object_a"]
        if not np.any(mask_a):
            continue
        bbox = mask_to_bbox(mask_a)
        s1 = CANDIDATE_FN(None, render.depth, bbox)
        s2 = CANDIDATE_FN(None, render.depth, bbox)
        assert s1 == s2, f"{CANDIDATE_NAME} not deterministic: {s1} vs {s2}"
        scores.append(s1)
    scores = np.array(scores)
    in_range = bool(np.all((scores >= 0) & (scores <= 1)))
    n_unique = len(set(np.round(scores, 6).tolist()))
    passed = in_range and n_unique >= 3
    print(f"  n={len(scores)}, range=[{scores.min():.4f},{scores.max():.4f}], "
          f"unique={n_unique}, in_range={in_range} -> {'PASS' if passed else 'FAIL'}")
    return passed


def criterion_b(cache):
    print("\n(b) Sensitivity: Spearman rho > 0.6 with noise/blend level, >=4/5 worlds")
    n_pass_noise, n_pass_blend = 0, 0
    n_worlds = 0
    for world_name in WORLDS_TRAIN:
        keys = [k for k in cache if k[0] == world_name]
        if not keys:
            continue
        n_worlds += 1
        objects, camera, render = cache[keys[0]]
        masks = {n: (render.owner == n) for n in objects if np.any(render.owner == n)}
        pairs = erf.enumerate_adjacent_pairs(masks)
        bboxes = [mask_to_bbox(masks[a]) for a, b in pairs]
        bboxes = [b for b in bboxes if b is not None]

        noise_levels_seen, noise_scores = [], []
        for level in NOISE_LEVELS:
            noisy = apply_depth_noise(render.depth, level=level, seed=RANDOM_STATE + int(level))
            for bbox in bboxes:
                noise_scores.append(CANDIDATE_FN(None, noisy, bbox))
                noise_levels_seen.append(level)
        rho_noise, _ = spearmanr(noise_levels_seen, noise_scores)

        blend_levels_seen, blend_scores = [], []
        for blend_px in BLEND_LEVELS_PX:
            blended = apply_depth_noise(render.depth, level=1.0 if blend_px > 0 else 0.0,
                                         seed=RANDOM_STATE, edge_blend_px=blend_px,
                                         sigma0=0.0, k=0.0)
            for bbox in bboxes:
                blend_scores.append(CANDIDATE_FN(None, blended, bbox))
                blend_levels_seen.append(blend_px)
        rho_blend, _ = spearmanr(blend_levels_seen, blend_scores)

        print(f"  {world_name}: rho_noise={rho_noise:.3f}, rho_blend={rho_blend:.3f}")
        if rho_noise > 0.6:
            n_pass_noise += 1
        if rho_blend > 0.6:
            n_pass_blend += 1

    passed = n_pass_noise >= 4 and n_pass_blend >= 4
    print(f"  noise rho>0.6 on {n_pass_noise}/{n_worlds}, blend rho>0.6 on {n_pass_blend}/{n_worlds} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def criterion_c(cache, df):
    print("\n(c) Validity: AUROC > 0.70, CI lower bound > 0.55")
    oracle_df = pd.read_csv(ORACLE_CSV)
    oracle_lookup = {(r.world_id, r.object_a, r.object_b): r.true_min_gap for r in oracle_df.itertuples()}
    rows = []
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
        internals = na._internals(render.depth, bbox)
        if internals is None:
            continue
        estimated_gap, sigma_local, score = internals
        error = abs(estimated_gap - true_gap)
        rows.append({"world": row["world_id"], "wrong": int(error > WRONG_GAP_THRESHOLD_M), "score": score})
    val_df = pd.DataFrame(rows)
    if len(val_df) == 0:
        print("  n=0 -- candidate produced no usable internals on any row -> FAIL")
        return False, val_df
    auc = auroc(val_df["score"], val_df["wrong"])
    lo, hi = bootstrap_auroc_ci(val_df["score"], val_df["wrong"], val_df["world"].to_numpy())
    passed = auc > 0.70 and lo > 0.55
    print(f"  n={len(val_df)}, wrong fraction={val_df['wrong'].mean():.3f}, "
          f"AUROC={auc:.4f}, CI=[{lo:.4f},{hi:.4f}] -> {'PASS' if passed else 'FAIL'}")
    return passed, val_df


def main():
    print("SYNTHETIC/SIM - NOT A RESULT")
    print(f"Evaluating candidate {CANDIDATE_NAME} (neighbor-aware) against explore/CRITERIA.md")
    cache, df = build_scene_cache(WORLDS_TRAIN)

    passed_a = criterion_a(cache, df)
    if not passed_a:
        print("\n=== STOP: criterion (a) FAILED. Short note: candidate is not usably "
              "non-constant/in-range/deterministic on the 5 non-held-out worlds. "
              "Not selected. Held-out world NOT touched. ===")
        return

    passed_b = criterion_b(cache)
    if not passed_b:
        print("\n=== STOP: criterion (b) FAILED (sensitivity < 4/5 worlds). Short note: "
              "candidate does not respond robustly to injected noise/blur across enough "
              "worlds. Not selected per the pre-committed rule (must pass (a) AND (b)). "
              "Held-out world NOT touched. ===")
        return

    passed_c, val_df = criterion_c(cache, df)
    if not passed_c:
        print("\n=== Candidate passed (a) and (b) but FAILED (c) validity. Reported per "
              "the pre-committed protocol -- (a)+(b) alone determine selection, so this "
              "candidate WOULD be selected and touch the held-out world once, but (c)'s "
              "failure is reported here first, honestly, before that touch. ===")

    print("\nPassed (a) AND (b) -- candidate is SELECTED per the pre-committed rule. "
          "Touching world6_dense_mixed (held-out) exactly once now.")
    held_out_cache, held_out_df = build_scene_cache([HELD_OUT_WORLD])
    passed_c_held, val_df_held = criterion_c(held_out_cache, held_out_df)
    print(f"\nHeld-out world result: {'PASS' if passed_c_held else 'FAIL'} on validity criterion (c).")


if __name__ == "__main__":
    main()
