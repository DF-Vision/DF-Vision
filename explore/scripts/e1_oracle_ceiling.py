#!/usr/bin/env python3
"""
explore/scripts/e1_oracle_ceiling.py

E1: oracle ceiling. Computes ground-truth geometric features (see
oracle_geometry.py -- independent of any depth/camera path) for every
(world, object_a, object_b) pair in the committed 330-row dataset
(retrain/data/relation_dataset_v2_SIM.csv), trains the same model
families (logistic regression, random forest) with the SAME leave-one-
world-out CV as the retrain session's G1/G2, and reports balanced
accuracy, per-class recall, bootstrap CI (grouped by world), and the
confusion matrix. world6_dense_mixed is held out and touched exactly
once, at the end, matching retrain/GATES.md's discipline.

Interpretation is per explore/CRITERIA.md's E1 rule, fixed before this
script was ever run.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "explore" / "scripts"))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import extract_relation_features as erf
from oracle_geometry import oracle_pair_features, ORACLE_FEATURE_COLUMNS

RANDOM_STATE = 20260920
DATA_CSV = PROJECT_ROOT / "retrain" / "data" / "relation_dataset_v2_SIM.csv"
OUT_CSV = PROJECT_ROOT / "explore" / "data" / "oracle_features_SIM.csv"
HELD_OUT_WORLD = "world6_dense_mixed"

CANDIDATES = {
    "logreg": lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=RANDOM_STATE, class_weight="balanced"),
}


def build_oracle_dataset():
    df = pd.read_csv(DATA_CSV)
    unique_pairs = df[["world_id", "object_a", "object_b"]].drop_duplicates()
    print(f"{len(unique_pairs)} unique (world,object_a,object_b) pairs across {df['world_id'].nunique()} worlds "
          f"(oracle features are viewpoint-independent -- computed once per pair, broadcast to all rows)")

    oracle_cache = {}
    world_objects = {}
    for world_name in df["world_id"].unique():
        objects, _ = erf.parse_world_sdf(erf.WORLDS_SDF_DIR / f"{world_name}.sdf")
        world_objects[world_name] = objects

    for _, row in unique_pairs.iterrows():
        w, a, b = row["world_id"], row["object_a"], row["object_b"]
        objects = world_objects[w]
        if a not in objects or b not in objects:
            continue
        feats = oracle_pair_features(objects[a], objects[b])
        oracle_cache[(w, a, b)] = feats

    rows = []
    for _, row in df.iterrows():
        key = (row["world_id"], row["object_a"], row["object_b"])
        feats = oracle_cache.get(key)
        if feats is None:
            continue
        out = {"world_id": row["world_id"], "viewpoint": row["viewpoint"],
               "object_a": row["object_a"], "object_b": row["object_b"],
               "relation": row["relation"]}
        out.update(feats)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out_df)} rows -> {OUT_CSV}")
    return out_df


def balanced_acc_and_recall(y_true, y_pred, classes):
    ba = balanced_accuracy_score(y_true, y_pred)
    recalls = {}
    for c in classes:
        mask = np.array(y_true) == c
        if mask.sum() > 0:
            recalls[c] = float((np.array(y_pred)[mask] == c).mean())
    return ba, recalls


def cv_evaluate(df, model_factory, n_splits):
    X = df[ORACLE_FEATURE_COLUMNS].to_numpy()
    y = df["relation"].to_numpy()
    groups = df["world_id"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    fold_scores = []
    all_preds = np.empty(len(df), dtype=object)
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe = make_pipeline(StandardScaler(), model_factory())
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        all_preds[test_idx] = pred
        fold_scores.append(balanced_accuracy_score(
            (y[test_idx] != "Separate").astype(int), (pred != "Separate").astype(int)))
    return np.array(fold_scores), all_preds


def bootstrap_ci_by_world(df, preds, n_boot=2000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    worlds = df["world_id"].unique()
    y_touch = (df["relation"] != "Separate").astype(int).to_numpy()
    pred_touch = (preds != "Separate").astype(int)
    accs = []
    for _ in range(n_boot):
        sampled = rng.choice(worlds, size=len(worlds), replace=True)
        idx = np.concatenate([np.where(df["world_id"].to_numpy() == w)[0] for w in sampled])
        if len(set(y_touch[idx])) < 2:
            continue
        accs.append(balanced_accuracy_score(y_touch[idx], pred_touch[idx]))
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main():
    full_df = build_oracle_dataset()
    train_df = full_df[full_df["world_id"] != HELD_OUT_WORLD].reset_index(drop=True)
    held_out_df = full_df[full_df["world_id"] == HELD_OUT_WORLD].reset_index(drop=True)

    n_worlds = train_df["world_id"].nunique()
    print(f"\nLeave-one-world-out CV: {n_worlds} folds over {sorted(train_df['world_id'].unique())}, "
          f"n={len(train_df)} rows (world6_dense_mixed excluded, n={len(held_out_df)})")
    print(f"Label counts (train): {train_df['relation'].value_counts().to_dict()}")

    for name, factory in CANDIDATES.items():
        print(f"\n=== oracle_{name} ===")
        scores, preds = cv_evaluate(train_df, factory, n_worlds)
        print(f"binary touching balanced accuracy: mean={scores.mean():.4f} std={scores.std():.4f} "
              f"folds={np.round(scores, 4).tolist()}")
        lo, hi = bootstrap_ci_by_world(train_df, preds)
        print(f"bootstrap 95% CI (grouped by world): [{lo:.4f}, {hi:.4f}]")

        y_true = train_df["relation"].to_numpy()
        for cls in ["Separate", "Contact", "Support"]:
            mask = y_true == cls
            if mask.sum() > 0:
                rec = float(np.mean(preds[mask] == cls))
                print(f"  recall({cls}) = {rec:.4f} (n={mask.sum()})")

        cm = confusion_matrix(y_true, preds, labels=["Separate", "Contact", "Support"])
        print("Confusion matrix (rows=true, cols=pred):")
        print(pd.DataFrame(cm, index=["Separate", "Contact", "Support"],
                            columns=["Separate", "Contact", "Support"]).to_string())

    print(f"\n{'='*78}\nHELD-OUT WORLD (world6_dense_mixed) -- touched once, right now\n{'='*78}")
    for name, factory in CANDIDATES.items():
        pipe = make_pipeline(StandardScaler(), factory())
        pipe.fit(train_df[ORACLE_FEATURE_COLUMNS].to_numpy(), train_df["relation"].to_numpy())
        pred_held = pipe.predict(held_out_df[ORACLE_FEATURE_COLUMNS].to_numpy())
        y_held = held_out_df["relation"].to_numpy()
        ba_held = balanced_accuracy_score((y_held != "Separate").astype(int), (pred_held != "Separate").astype(int))
        print(f"oracle_{name} on held-out world6 (n={len(held_out_df)}): binary balanced accuracy = {ba_held:.4f}")
        for cls in ["Separate", "Contact", "Support"]:
            mask = y_held == cls
            if mask.sum() > 0:
                rec = float(np.mean(pred_held[mask] == cls))
                print(f"  recall({cls}) = {rec:.4f} (n={mask.sum()})")

    print("\n--- True gap distribution, Contact vs Separate (for the CRITERIA.md interpretation rule) ---")
    for cls in ["Separate", "Contact", "Support"]:
        vals = full_df[full_df["relation"] == cls]["true_min_gap"]
        print(f"{cls}: n={len(vals)}, mean={vals.mean():.5f}m, std={vals.std():.5f}m, "
              f"min={vals.min():.5f}m, max={vals.max():.5f}m, "
              f"median={vals.median():.5f}m")


if __name__ == "__main__":
    main()
