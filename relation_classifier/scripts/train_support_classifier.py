"""
scripts/train_support_classifier.py

Train and persist the Support-vs-Contact sub-classifier that replaces
classify_relation.py's SUPPORT_RELATIVE_HEIGHT_THRESHOLD approach.

Only touching pairs (heuristic_label in {Contact, Support}) are used --
Separate is out of scope, since this sub-decision never runs on Separate
pairs in production (classify_relation.py's binary touching model filters
those out first).

relative_height is fed in as its absolute value: object_a/object_b column
ordering in the training data is set by enumerate_adjacent_pairs()'s SDF
document order, not physical stacking order, so raw signed relative_height
does not consistently mean "target below neighbor" (confirmed: 8/75
Support rows have positive raw relative_height). classify_relation.py's
inference-time feature construction must apply the same abs().

Model choice (signed off): Logistic Regression, C=1.0 -- not Random
Forest, despite RF's higher mean F1/balanced accuracy in cross-validation
(McNemar's p=0.50, not statistically distinguishable) -- because LogReg's
much lower fold-to-fold variance (std 0.06-0.09 vs RF's 0.13-0.14) makes
it the safer choice for production on unseen real-hardware object pairs,
and it is more interpretable for the paper's methods/limitations section.

Group-aware 5-fold CV (world_id, object_a, object_b), matching this
project's existing precedent (see relation_classifier/models/summary.txt).
Reported here for documentation only -- the persisted model is fit on
ALL 176 rows, not a train/test subset, per explicit instruction (unlike
train_relation_classifier.py's own saved model, which is fit on only its
train split).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS: list[str] = ["minimum_boundary_distance", "overlap_ratio", "relative_height"]
LOGREG_C = 1.0  # signed off, see module docstring
N_SPLITS = 5
RANDOM_STATE = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/training_data_metric_only.csv",
                         help="Path to the training data CSV (relative to relation_classifier/).")
    parser.add_argument("--output-dir", type=str, default="models/",
                         help="Directory to save the trained model, scaler, and summary.")
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df[df["heuristic_label"].isin(["Contact", "Support"])].copy()
    print(f"Evaluation/training set: {len(df)} rows "
          f"(Contact={sum(df.heuristic_label=='Contact')}, Support={sum(df.heuristic_label=='Support')})")
    return df


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[FEATURE_COLUMNS].to_numpy().astype(float)
    X[:, FEATURE_COLUMNS.index("relative_height")] = np.abs(X[:, FEATURE_COLUMNS.index("relative_height")])
    y = (df["heuristic_label"] == "Support").astype(int).to_numpy()
    groups = (df["world_id"] + "::" + df["object_a"] + "::" + df["object_b"]).to_numpy()
    return X, y, groups


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    X, y, groups = build_features(df)

    summary_blocks: list[str] = []
    summary_blocks.append(f"Data source: {args.data}\n")
    summary_blocks.append(f"Rows: {len(df)} (Contact={sum(y==0)}, Support={sum(y==1)}), "
                           f"unique groups: {len(set(groups))}\n")
    summary_blocks.append(f"Features: {FEATURE_COLUMNS} (relative_height fed as abs() -- see module docstring)\n")
    summary_blocks.append(f"Model: LogisticRegression(C={LOGREG_C}), StandardScaler-preprocessed\n\n")

    # --- Documentation-only group-aware CV (final model below is fit on all rows) ---
    gkf = GroupKFold(n_splits=N_SPLITS)
    f1s, baccs = [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=LOGREG_C, max_iter=2000, random_state=RANDOM_STATE))
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        f1s.append(f1_score(y[test_idx], pred, pos_label=1, zero_division=0))
        baccs.append(balanced_accuracy_score(y[test_idx], pred))
    f1s, baccs = np.array(f1s), np.array(baccs)
    cv_line = (f"Group-aware {N_SPLITS}-fold CV (documentation only, NOT the persisted model): "
               f"F1(Support) mean={f1s.mean():.4f} std={f1s.std():.4f} folds={np.round(f1s,4).tolist()}\n"
               f"  balanced_accuracy mean={baccs.mean():.4f} std={baccs.std():.4f} folds={np.round(baccs,4).tolist()}\n")
    print(cv_line)
    summary_blocks.append(cv_line + "\n")

    baseline_line = ("Comparison baselines (from the prior threshold sweep, same CV methodology): "
                      "old threshold -0.01 (deployed) F1=0.3575/bal_acc=0.3616; "
                      "best single-feature threshold -0.002375 F1=0.5972/bal_acc=0.5288; "
                      "always-Support degenerate rule F1=0.5914/bal_acc=0.5000.\n"
                      "McNemar's test vs. old threshold approaches: p<0.0001 (both), i.e. this model's error "
                      "pattern is statistically distinguishable from the old threshold's, not a relabeling of "
                      "the same mistakes. RF was also evaluated (F1=0.7302/bal_acc=0.7587, both higher than "
                      "LogReg) but McNemar's RF-vs-LogReg p=0.4996 -- not statistically distinguishable from "
                      "LogReg, so LogReg's much lower fold variance (std 0.06-0.09 vs RF's 0.13-0.14) was the "
                      "deciding factor for production deployment on unseen real-hardware object pairs.\n")
    print(baseline_line)
    summary_blocks.append(baseline_line + "\n")

    # --- Final model: fit on ALL rows, this is what gets persisted ---
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    final_model = LogisticRegression(C=LOGREG_C, max_iter=2000, random_state=RANDOM_STATE)
    final_model.fit(X_scaled, y)

    coef_lines = ["Logistic Regression coefficients (standardized scale, final model fit on all rows):"]
    order = np.argsort(-np.abs(final_model.coef_[0]))
    for idx in order:
        coef_lines.append(f"  {FEATURE_COLUMNS[idx]:<28s} {final_model.coef_[0][idx]:+.4f}")
    coef_block = "\n".join(coef_lines) + "\n"
    print(coef_block)
    summary_blocks.append(coef_block)

    model_path = output_dir / "support_logistic_model.joblib"
    scaler_path = output_dir / "support_scaler.joblib"
    joblib.dump(final_model, model_path)
    joblib.dump(final_scaler, scaler_path)
    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")

    summary_path = output_dir / "support_summary.txt"
    with open(summary_path, "w") as f:
        f.write("".join(summary_blocks))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
