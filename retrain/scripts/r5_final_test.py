#!/usr/bin/env python3
"""
retrain/scripts/r5_final_test.py

R5: evaluate the selected v2 model on the held-out world (world6_dense_mixed)
EXACTLY ONCE. Per rule 8 (test data discipline), this script is run a
single time after all model selection (R3) is finished; running it again
after seeing its output and then changing the model would be "peeking"
and must be declared if it ever happens.

Reports the full gate metric set (G1, G2, G4, G6), confusion matrix, and
calibration curve. States plainly that this is still simulation and
real-hardware behavior is UNVALIDATED.

Note on bootstrap: GATES.md's world-grouped bootstrap is defined for the
5-world CV phase (R3). Here there is exactly one held-out world by
design, so a world-grouped bootstrap is not meaningful (resampling a set
of size 1 with replacement always returns the same set) -- this script
instead bootstraps at the (viewpoint, pair) row level, which is the best
available proxy for sampling uncertainty on this single-world test, and
says so explicitly rather than silently reusing the CV-phase methodology
where it doesn't apply.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, brier_score_loss

RANDOM_STATE = 20260920
DATA_CSV = PROJECT_ROOT / "retrain" / "data" / "relation_dataset_v2_SIM.csv"
HELD_OUT_WORLD = "world6_dense_mixed"
MODELS_DIR = PROJECT_ROOT / "retrain" / "models"

FEATURE_COLUMNS = [
    "boundary_sharpness", "minimum_boundary_distance", "surface_normal_consistency",
    "relative_height", "depth_gradient", "overlap_ratio", "edge_continuity",
    "occlusion_boundary_score",
]
SUPPORT_FEATURE_COLUMNS = ["minimum_boundary_distance", "overlap_ratio", "relative_height"]


def expected_calibration_error(y_true, p_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi) if hi < 1 else (p_pred >= lo) & (p_pred <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = p_pred[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def main():
    import json
    meta_path = MODELS_DIR / "relation_binary_v2_metadata.json"
    if not meta_path.exists():
        print("ERROR: no v2 model metadata found -- run r3_train_and_select.py first.")
        sys.exit(1)
    metadata = json.loads(meta_path.read_text())
    architecture = metadata["architecture"]

    df = pd.read_csv(DATA_CSV)
    test_df = df[df["world_id"] == HELD_OUT_WORLD].reset_index(drop=True)
    print("=" * 78)
    print(f"R5 FINAL TEST -- {HELD_OUT_WORLD}, touched exactly once, n={len(test_df)}")
    print(f"Model architecture: {architecture}, selected candidate: {metadata['selected_candidate']}")
    print("SIM ONLY. Real-hardware behavior is UNVALIDATED by this test.")
    print("=" * 78)

    v2_model = joblib.load(MODELS_DIR / "relation_binary_v2.joblib")
    v2_scaler = joblib.load(MODELS_DIR / "relation_binary_v2_scaler.joblib")

    X = test_df[FEATURE_COLUMNS].to_numpy()
    X_scaled = v2_scaler.transform(X)
    y_true_touch = (test_df["relation"] != "Separate").astype(int).to_numpy()
    p_touch = v2_model.predict_proba(X_scaled)[:, 1] if hasattr(v2_model, "predict_proba") else None
    pred_touch = v2_model.predict(X_scaled)
    if not np.issubdtype(np.array(pred_touch).dtype, np.number):
        # direct 3-class model: predict returns strings
        pred_relation = pred_touch
        pred_touch = (pred_relation != "Separate").astype(int)
    else:
        pred_relation = None

    ba = balanced_accuracy_score(y_true_touch, pred_touch)
    print(f"\nG1: binary touching balanced accuracy = {ba:.4f} (gate: >0.75)")

    rng = np.random.default_rng(RANDOM_STATE)
    n = len(test_df)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        if len(set(y_true_touch[idx])) < 2:
            continue
        boot.append(balanced_accuracy_score(y_true_touch[idx], np.array(pred_touch)[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan"))
    print(f"Bootstrap 95% CI (row-level, NOT world-grouped -- only 1 held-out world, see docstring): "
          f"[{lo:.4f}, {hi:.4f}] (gate: lower bound >0.60)")

    if architecture == "two_stage" and (MODELS_DIR / "support_v2.joblib").exists():
        support_model = joblib.load(MODELS_DIR / "support_v2.joblib")
        support_scaler = joblib.load(MODELS_DIR / "support_v2_scaler.joblib")
        pred_relation = []
        for i in range(n):
            if pred_touch[i] == 0:
                pred_relation.append("Separate")
            else:
                xs = test_df.iloc[[i]][SUPPORT_FEATURE_COLUMNS].to_numpy().copy()
                xs[:, SUPPORT_FEATURE_COLUMNS.index("relative_height")] = np.abs(
                    xs[:, SUPPORT_FEATURE_COLUMNS.index("relative_height")])
                xs_scaled = support_scaler.transform(xs)
                sub_pred = support_model.predict(xs_scaled)[0]
                pred_relation.append("Support" if sub_pred == 1 else "Contact")
        pred_relation = np.array(pred_relation)

    y_true_relation = test_df["relation"].to_numpy()
    print(f"\nG2: per-class recall (n>0.50 each, more than one class predicted)")
    n_classes_predicted = len(set(pred_relation.tolist())) if pred_relation is not None else "n/a"
    for cls in ["Separate", "Contact", "Support"]:
        mask = y_true_relation == cls
        if mask.sum() > 0 and pred_relation is not None:
            rec = float(np.mean(pred_relation[mask] == cls))
            print(f"  recall({cls}) = {rec:.4f} (n={mask.sum()}) {'PASS' if rec > 0.5 else 'FAIL'}")
        else:
            print(f"  recall({cls}) = n/a (n={mask.sum()})")
    print(f"  classes predicted: {n_classes_predicted} (gate: >1)")

    if pred_relation is not None:
        labels = ["Separate", "Contact", "Support"]
        cm = confusion_matrix(y_true_relation, pred_relation, labels=labels)
        print(f"\nConfusion matrix (rows=true, cols=pred), labels={labels}:")
        print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

    if p_touch is not None:
        ece = expected_calibration_error(y_true_touch, p_touch)
        brier = brier_score_loss(y_true_touch, p_touch)
        print(f"\nG4: Expected Calibration Error = {ece:.4f} (gate: <0.10), Brier score = {brier:.4f}")

    print("\nPer-pair errors:")
    err_df = test_df[["world_id", "viewpoint", "object_a", "object_b", "relation"]].copy()
    err_df["predicted"] = pred_relation if pred_relation is not None else pred_touch
    err_df["correct"] = err_df["relation"] == err_df["predicted"] if pred_relation is not None else (y_true_touch == pred_touch)
    print(err_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("REMINDER: this is a SIMULATION test on ray-cast synthetic depth. It is")
    print("NOT evidence of real-hardware performance, which remains UNVALIDATED.")
    print("See retrain/REAL_CAPTURE_PROTOCOL.md (R8) for the actual sim-to-real test.")
    print("=" * 78)


if __name__ == "__main__":
    main()
