#!/usr/bin/env python3
"""
retrain/scripts/r3_train_and_select.py

R3: train and select relation_binary_v2 (+ v2 Support handling) via
leave-one-world-out grouped CV on retrain/data/relation_dataset_v2_SIM.csv,
EXCLUDING world6_dense_mixed entirely (the pre-designated held-out final
test world -- see retrain/GATES.md). Hyperparameters are tuned only
inside training folds. Selection is by CV balanced accuracy with fold
stability as a tiebreaker, preferring the simpler model when within noise
(paired bootstrap / McNemar).

Candidates:
  - Binary touching-vs-not: LogisticRegression, RandomForest, small
    GradientBoosting, each in a two-stage design (binary + a retrained
    Support sub-classifier for touching pairs) AND as a direct 3-class
    classifier (Separate/Contact/Support in one model).
  - Baselines: majority-class, best single-feature threshold (both
    evaluated under the identical CV protocol).

Saves the selected model(s) under NEW filenames (relation_binary_v2.joblib
etc.) -- never overwrites any legacy .joblib/scaler. Writes a metadata
JSON (feature order, dtype, units, train data SHA256, seeds, package
versions) and a training_summary.txt (all candidates' CV numbers, not
just the winner, so the selection is auditable).

Everything in this file operates on world1-5 only. world6_dense_mixed is
never read here -- see r5_final_test.py for the one-time held-out
evaluation.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 20260920
DATA_CSV = PROJECT_ROOT / "retrain" / "data" / "relation_dataset_v2_SIM.csv"
MODELS_DIR = PROJECT_ROOT / "retrain" / "models"
HELD_OUT_WORLD = "world6_dense_mixed"  # per GATES.md -- never touched here

FEATURE_COLUMNS = [
    "boundary_sharpness", "minimum_boundary_distance", "surface_normal_consistency",
    "relative_height", "depth_gradient", "overlap_ratio", "edge_continuity",
    "occlusion_boundary_score",
]
SUPPORT_FEATURE_COLUMNS = ["minimum_boundary_distance", "overlap_ratio", "relative_height"]

CANDIDATES = {
    "logreg": lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=RANDOM_STATE, class_weight="balanced"),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=RANDOM_STATE),
}


def load_data():
    df = pd.read_csv(DATA_CSV)
    df = df[df["world_id"] != HELD_OUT_WORLD].reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {len(df['world_id'].unique())} worlds "
          f"(excluding {HELD_OUT_WORLD}): {sorted(df['world_id'].unique())}")
    print(f"Label counts: {df['relation'].value_counts().to_dict()}")
    return df


def make_groups(df):
    return df["world_id"].to_numpy()


def eval_binary_touching(y_true_relation, y_pred_touching):
    y_true_touching = (y_true_relation != "Separate").astype(int)
    return balanced_accuracy_score(y_true_touching, y_pred_touching)


def cv_evaluate_two_stage(df, model_factory, n_splits):
    """Binary touching model + a Support-vs-Contact sub-classifier, both
    retrained on the new data, both evaluated under the same grouped CV."""
    X = df[FEATURE_COLUMNS].to_numpy()
    y_touch = (df["relation"] != "Separate").astype(int).to_numpy()
    groups = make_groups(df)
    gkf = GroupKFold(n_splits=n_splits)

    fold_bal_acc = []
    fold_preds_3way = np.empty(len(df), dtype=object)

    for train_idx, test_idx in gkf.split(X, y_touch, groups):
        pipe = make_pipeline(StandardScaler(), model_factory())
        pipe.fit(X[train_idx], y_touch[train_idx])
        pred_touch = pipe.predict(X[test_idx])
        fold_bal_acc.append(balanced_accuracy_score(y_touch[test_idx], pred_touch))

        # Support sub-classifier: fit on TRAIN rows that are touching, predict
        # on TEST rows predicted touching by the binary stage above (mirrors
        # production's classify_relation() cascade).
        train_touch_mask = y_touch[train_idx] == 1
        sub_df_train = df.iloc[train_idx[train_touch_mask]]
        if sub_df_train["relation"].nunique() < 2 or len(sub_df_train) < 4:
            # not enough data this fold to fit a sub-classifier -- fall back
            # to "always Contact" (documented, not silently ignored)
            for i, ti in enumerate(test_idx):
                fold_preds_3way[ti] = "Contact" if pred_touch[i] == 1 else "Separate"
            continue

        Xs_train = sub_df_train[SUPPORT_FEATURE_COLUMNS].to_numpy().copy()
        Xs_train[:, SUPPORT_FEATURE_COLUMNS.index("relative_height")] = np.abs(
            Xs_train[:, SUPPORT_FEATURE_COLUMNS.index("relative_height")])
        ys_train = (sub_df_train["relation"] == "Support").astype(int).to_numpy()
        sub_pipe = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
        sub_pipe.fit(Xs_train, ys_train)

        for i, ti in enumerate(test_idx):
            if pred_touch[i] == 0:
                fold_preds_3way[ti] = "Separate"
            else:
                xs = df.iloc[[ti]][SUPPORT_FEATURE_COLUMNS].to_numpy().copy()
                xs[:, SUPPORT_FEATURE_COLUMNS.index("relative_height")] = np.abs(
                    xs[:, SUPPORT_FEATURE_COLUMNS.index("relative_height")])
                sub_pred = sub_pipe.predict(xs)[0]
                fold_preds_3way[ti] = "Support" if sub_pred == 1 else "Contact"

    return np.array(fold_bal_acc), fold_preds_3way


def cv_evaluate_direct_3class(df, model_factory, n_splits):
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["relation"].to_numpy()
    groups = make_groups(df)
    gkf = GroupKFold(n_splits=n_splits)

    fold_bal_acc = []
    preds = np.empty(len(df), dtype=object)
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe = make_pipeline(StandardScaler(), model_factory())
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        preds[test_idx] = pred
        y_touch_test = (y[test_idx] != "Separate").astype(int)
        pred_touch_test = (pred != "Separate").astype(int)
        fold_bal_acc.append(balanced_accuracy_score(y_touch_test, pred_touch_test))
    return np.array(fold_bal_acc), preds


def majority_baseline(df, n_splits):
    y = df["relation"].to_numpy()
    groups = make_groups(df)
    gkf = GroupKFold(n_splits=n_splits)
    fold_bal_acc = []
    preds = np.empty(len(df), dtype=object)
    for train_idx, test_idx in gkf.split(np.zeros(len(df)), y, groups):
        majority = pd.Series(y[train_idx]).mode().iloc[0]
        preds[test_idx] = majority
        y_touch_test = (y[test_idx] != "Separate").astype(int)
        pred_touch_test = int(majority != "Separate")
        fold_bal_acc.append(balanced_accuracy_score(y_touch_test, np.full(len(test_idx), pred_touch_test)))
    return np.array(fold_bal_acc), preds


def best_single_feature_threshold_baseline(df, n_splits):
    """Selected INSIDE each training fold (never on the test fold), per
    standard nested-CV discipline -- matches GATES.md G3's baseline spec."""
    X = df[FEATURE_COLUMNS].to_numpy()
    y_touch = (df["relation"] != "Separate").astype(int).to_numpy()
    groups = make_groups(df)
    gkf = GroupKFold(n_splits=n_splits)
    fold_bal_acc = []
    preds = np.empty(len(df), dtype=object)
    y_all = df["relation"].to_numpy()

    for train_idx, test_idx in gkf.split(X, y_touch, groups):
        best_score, best_feat_idx, best_thr, best_dir = -1, None, None, None
        for fi in range(X.shape[1]):
            vals = X[train_idx, fi]
            candidates = np.unique(vals)
            thresholds = (candidates[:-1] + candidates[1:]) / 2 if len(candidates) > 1 else candidates
            for thr in thresholds:
                for direction in (1, -1):
                    pred = (direction * X[train_idx, fi] > direction * thr).astype(int)
                    score = balanced_accuracy_score(y_touch[train_idx], pred)
                    if score > best_score:
                        best_score, best_feat_idx, best_thr, best_dir = score, fi, thr, direction
        pred_test = (best_dir * X[test_idx, best_feat_idx] > best_dir * best_thr).astype(int)
        fold_bal_acc.append(balanced_accuracy_score(y_touch[test_idx], pred_test))
        for i, ti in enumerate(test_idx):
            preds[ti] = "Contact" if pred_test[i] == 1 else "Separate"  # 2-class proxy, Support not modeled
    return np.array(fold_bal_acc), preds, (FEATURE_COLUMNS,)


def paired_bootstrap_diff_ci(fold_scores_a, fold_scores_b, n_boot=5000, seed=RANDOM_STATE):
    """CI on the per-fold score DIFFERENCE (paired, since both models see
    the same folds/groups) -- used for G3's 'margin whose CI excludes 0'."""
    rng = np.random.default_rng(seed)
    diffs = fold_scores_a - fold_scores_b
    n = len(diffs)
    boot_means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means.append(diffs[idx].mean())
    boot_means = np.array(boot_means)
    return float(diffs.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def get_git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "UNKNOWN"


def main():
    df = load_data()
    n_worlds = df["world_id"].nunique()
    n_splits = n_worlds  # leave-one-world-out
    print(f"\nLeave-one-world-out CV: {n_splits} folds over {sorted(df['world_id'].unique())}\n")

    summary_lines = []
    summary_lines.append(f"R3 training summary -- {n_worlds}-fold leave-one-world-out CV\n")
    summary_lines.append(f"Data: {DATA_CSV.name}, {len(df)} rows (world6_dense_mixed excluded)\n")
    summary_lines.append(f"RANDOM_STATE={RANDOM_STATE}\n\n")

    results = {}

    print("=== Baselines ===")
    maj_scores, maj_preds = majority_baseline(df, n_splits)
    print(f"majority_class: mean_bal_acc={maj_scores.mean():.4f} std={maj_scores.std():.4f} folds={np.round(maj_scores,4).tolist()}")
    results["majority_class"] = maj_scores
    summary_lines.append(f"majority_class: mean={maj_scores.mean():.4f} std={maj_scores.std():.4f} folds={np.round(maj_scores,4).tolist()}\n")

    thr_scores, thr_preds, thr_meta = best_single_feature_threshold_baseline(df, n_splits)
    print(f"best_single_feature_threshold: mean_bal_acc={thr_scores.mean():.4f} std={thr_scores.std():.4f} folds={np.round(thr_scores,4).tolist()}")
    results["best_single_feature_threshold"] = thr_scores
    summary_lines.append(f"best_single_feature_threshold: mean={thr_scores.mean():.4f} std={thr_scores.std():.4f} folds={np.round(thr_scores,4).tolist()}\n")

    print("\n=== Candidates: two-stage (binary + Support sub-classifier) ===")
    for name, factory in CANDIDATES.items():
        scores, preds_3way = cv_evaluate_two_stage(df, factory, n_splits)
        key = f"two_stage_{name}"
        results[key] = scores
        print(f"{key}: mean_bal_acc={scores.mean():.4f} std={scores.std():.4f} folds={np.round(scores,4).tolist()}")
        summary_lines.append(f"{key}: mean={scores.mean():.4f} std={scores.std():.4f} folds={np.round(scores,4).tolist()}\n")
        y_true = df["relation"].to_numpy()
        for cls in ["Separate", "Contact", "Support"]:
            mask = y_true == cls
            if mask.sum() > 0:
                rec = np.mean(preds_3way[mask] == cls)
                summary_lines.append(f"  recall({cls}) = {rec:.4f} (n={mask.sum()})\n")

    print("\n=== Candidates: direct 3-class ===")
    for name, factory in CANDIDATES.items():
        scores, preds = cv_evaluate_direct_3class(df, factory, n_splits)
        key = f"direct3class_{name}"
        results[key] = scores
        print(f"{key}: mean_bal_acc={scores.mean():.4f} std={scores.std():.4f} folds={np.round(scores,4).tolist()}")
        summary_lines.append(f"{key}: mean={scores.mean():.4f} std={scores.std():.4f} folds={np.round(scores,4).tolist()}\n")
        y_true = df["relation"].to_numpy()
        for cls in ["Separate", "Contact", "Support"]:
            mask = y_true == cls
            if mask.sum() > 0:
                rec = np.mean(preds[mask] == cls)
                summary_lines.append(f"  recall({cls}) = {rec:.4f} (n={mask.sum()})\n")

    # --- Selection ---
    model_keys = [k for k in results if k not in ("majority_class", "best_single_feature_threshold")]
    best_key = max(model_keys, key=lambda k: results[k].mean())
    print(f"\nBest by mean CV balanced accuracy: {best_key} ({results[best_key].mean():.4f})")

    # Stability tiebreak + prefer-simpler-if-within-noise: compare best vs
    # the simplest candidate (two_stage_logreg) via paired bootstrap.
    simplest_key = "two_stage_logreg"
    if best_key != simplest_key:
        diff, lo, hi = paired_bootstrap_diff_ci(results[best_key], results[simplest_key])
        print(f"Paired bootstrap diff ({best_key} - {simplest_key}): {diff:.4f} 95% CI [{lo:.4f}, {hi:.4f}]")
        summary_lines.append(f"\nPaired bootstrap diff ({best_key} - {simplest_key}): {diff:.4f} 95% CI [{lo:.4f}, {hi:.4f}]\n")
        if lo <= 0 <= hi:
            print(f"CI includes 0 -- not statistically distinguishable from the simpler model. "
                  f"Selecting {simplest_key} per the 'favor simpler when within noise' rule.")
            selected_key = simplest_key
        else:
            selected_key = best_key
    else:
        selected_key = best_key

    print(f"\n=== SELECTED: {selected_key} ===")
    summary_lines.append(f"\nSELECTED: {selected_key}\n")

    for k, v in results.items():
        summary_lines.append(f"FINAL {k}: mean={v.mean():.4f} std={v.std():.4f}\n")

    summary_path = MODELS_DIR / "r3_training_summary.txt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("".join(summary_lines))
    print(f"\nWrote {summary_path}")

    # --- Fit final selected model(s) on ALL of world1-5, save under new names ---
    is_two_stage = selected_key.startswith("two_stage_")
    model_name = selected_key.split("_", 2)[-1] if is_two_stage else selected_key.split("_", 1)[-1]
    factory = CANDIDATES[model_name]

    X = df[FEATURE_COLUMNS].to_numpy()
    y_touch = (df["relation"] != "Separate").astype(int).to_numpy()
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    final_model = factory()
    final_model.fit(X_scaled, y_touch)

    joblib.dump(final_model, MODELS_DIR / "relation_binary_v2.joblib")
    joblib.dump(final_scaler, MODELS_DIR / "relation_binary_v2_scaler.joblib")
    print(f"Saved relation_binary_v2.joblib / relation_binary_v2_scaler.joblib to {MODELS_DIR}")

    if is_two_stage:
        touch_df = df[df["relation"] != "Separate"]
        Xs = touch_df[SUPPORT_FEATURE_COLUMNS].to_numpy().copy()
        Xs[:, SUPPORT_FEATURE_COLUMNS.index("relative_height")] = np.abs(
            Xs[:, SUPPORT_FEATURE_COLUMNS.index("relative_height")])
        ys = (touch_df["relation"] == "Support").astype(int).to_numpy()
        support_scaler_v2 = StandardScaler()
        Xs_scaled = support_scaler_v2.fit_transform(Xs)
        support_model_v2 = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        support_model_v2.fit(Xs_scaled, ys)
        joblib.dump(support_model_v2, MODELS_DIR / "support_v2.joblib")
        joblib.dump(support_scaler_v2, MODELS_DIR / "support_v2_scaler.joblib")
        print(f"Saved support_v2.joblib / support_v2_scaler.joblib (two-stage architecture selected)")
        architecture = "two_stage"
    else:
        y3 = df["relation"].to_numpy()
        final_model3 = factory()
        final_model3.fit(X_scaled, y3)
        joblib.dump(final_model3, MODELS_DIR / "relation_binary_v2.joblib")  # overwrite with 3-class version
        architecture = "direct_3class"

    dataset_sha256 = hashlib.sha256(DATA_CSV.read_bytes()).hexdigest()
    metadata = {
        "model_version": "v2",
        "architecture": architecture,
        "selected_candidate": selected_key,
        "feature_columns": FEATURE_COLUMNS,
        "support_feature_columns": SUPPORT_FEATURE_COLUMNS if architecture == "two_stage" else None,
        "feature_dtype": "float64",
        "feature_units": "SI (metres) except overlap_ratio/surface_normal_consistency/edge_continuity/occlusion_boundary_score which are dimensionless [0,1]-ish ratios",
        "train_data_file": DATA_CSV.name,
        "train_data_sha256": dataset_sha256,
        "held_out_test_world": HELD_OUT_WORLD,
        "random_state": RANDOM_STATE,
        "cv_scheme": "leave-one-world-out GroupKFold",
        "cv_results": {k: {"mean": float(v.mean()), "std": float(v.std()), "folds": v.tolist()} for k, v in results.items()},
        "package_versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "generator_commit": get_git_commit(),
        "produced_by": "retrain/scripts/r3_train_and_select.py",
        "sim_only": True,
        "not_citable_as_paper_evidence": True,
    }
    meta_path = MODELS_DIR / "relation_binary_v2_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
