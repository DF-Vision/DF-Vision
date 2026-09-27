#!/usr/bin/env python3
"""
project_audit/scripts/paired_comparison.py

C4 audit evidence generator (read-only -- imports and runs existing
committed code, never writes to it). Computes C1's estimated_gap and
AUROC using BOTH the pre-fix (straddling ring, reconstructed from git
history via prefix_c1_internals.py) and post-fix (exterior-only ring,
the current committed code) implementations on the EXACT SAME set of
(world, object_a, object_b, viewpoint) rows, then does a paired,
world-grouped bootstrap on the AUROC DIFFERENCE (post - pre) to test
whether the change is distinguishable from noise -- a stricter test than
checking whether one point estimate falls inside the other's marginal CI.
Exact output quoted in project_audit/C4_AUDIT.md section 2.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "explore" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "retrain" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import geometric_prg_error as gpe_postfix
from prefix_c1_internals import _c1_internals_prefix
from e3_evaluate_candidates import build_scene_cache, mask_to_bbox, auroc, WORLDS_TRAIN, WRONG_GAP_THRESHOLD_M, ORACLE_CSV, RANDOM_STATE

cache, df = build_scene_cache(WORLDS_TRAIN)
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

    pre = _c1_internals_prefix(render.depth, bbox)
    post = gpe_postfix._c1_internals(render.depth, bbox)
    if pre is None or post is None:
        continue

    pre_gap, _, pre_score = pre
    post_gap, _, post_score = post
    pre_wrong = int(abs(pre_gap - true_gap) > WRONG_GAP_THRESHOLD_M)
    post_wrong = int(abs(post_gap - true_gap) > WRONG_GAP_THRESHOLD_M)
    rows.append({
        "world": row["world_id"], "pre_score": pre_score, "post_score": post_score,
        "pre_wrong": pre_wrong, "post_wrong": post_wrong,
    })

val_df = pd.DataFrame(rows)
print(f"n={len(val_df)} rows with BOTH pre-fix and post-fix internals available")
print(f"pre-fix wrong fraction={val_df['pre_wrong'].mean():.4f} "
      f"(committed pre-fix run reported 0.737 on n=262 -- compare)")
print(f"post-fix wrong fraction={val_df['post_wrong'].mean():.4f} "
      f"(committed post-fix run reported 0.821 on n=262 -- compare)")

auc_pre = auroc(val_df["pre_score"], val_df["pre_wrong"])
auc_post = auroc(val_df["post_score"], val_df["post_wrong"])
print(f"\nAUROC pre-fix (this re-derivation)  = {auc_pre:.4f}  (committed: 0.4567)")
print(f"AUROC post-fix (this re-derivation) = {auc_post:.4f}  (committed: 0.2796)")

# Paired, world-grouped bootstrap on the DIFFERENCE (post - pre), using
# the SAME resampled world set for both computations in each resample.
rng = np.random.default_rng(RANDOM_STATE)
unique_worlds = val_df["world"].unique()
diffs = []
for _ in range(2000):
    sampled = rng.choice(unique_worlds, size=len(unique_worlds), replace=True)
    idx = np.concatenate([np.where(val_df["world"].to_numpy() == w)[0] for w in sampled])
    pre_wrong_s = val_df["pre_wrong"].to_numpy()[idx]
    post_wrong_s = val_df["post_wrong"].to_numpy()[idx]
    if len(set(pre_wrong_s.tolist())) < 2 or len(set(post_wrong_s.tolist())) < 2:
        continue
    a_pre = auroc(val_df["pre_score"].to_numpy()[idx], pre_wrong_s)
    a_post = auroc(val_df["post_score"].to_numpy()[idx], post_wrong_s)
    diffs.append(a_post - a_pre)

diffs = np.array(diffs)
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"\nPaired world-grouped bootstrap (n_boot_used={len(diffs)}/2000) on AUROC(post) - AUROC(pre):")
print(f"  mean diff = {diffs.mean():.4f}")
print(f"  95% CI    = [{lo:.4f}, {hi:.4f}]")
print(f"  CI excludes 0 (i.e. change is distinguishable from noise): {'YES' if (lo > 0 or hi < 0) else 'NO'}")
