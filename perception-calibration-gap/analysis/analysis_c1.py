"""
analysis_c1.py

Purpose
-------
Test whether Gap adds predictive power for failure beyond true_occlusion
alone, controlling for object identity (the pre-registered generalization
claim). See PREREG.md Section 2, C1.

Pre-registered model forms:
    Model A: failure ~ true_occlusion + C(object_id)
    Model B: failure ~ true_occlusion + Gap_initial + C(object_id)
    Model C: failure ~ true_occlusion + Gap_max + C(object_id)

Pre-registered pass criteria (do not loosen these after seeing results):
    1. Gap coefficient significant (p<0.05) in Model B or C, odds-ratio CI excludes 1
    2. Delta-AUC (B or C vs A) positive, bootstrap CI excludes 0
    3. Random Forest + SHAP ranks Gap_max or Gap_initial in top-2 features
    4. VIF(Gap, true_occlusion) checked; flagged if > 10
    5. Per-object descriptive table: Gap coefficient sign consistent across
       objects (this -- not per-object significance -- is the generalization
       evidence, since ~30-50 trials/object is underpowered per-object)

How to check this script is correct (before trusting it on real data):
    Run it against mock_data_generator.py's output, where the ground-truth
    Gap coefficient (TRUE_COEFS['b_gap'] = 2.5) is known. The recovered
    coefficient should be positive and roughly in that ballpark, and should
    be significant. If it comes back negative or null on the mock data,
    the script has a bug -- fix it before ever pointing it at real trials.

Edge cases handled:
    - object_id is passed through C() as a categorical fixed effect, not
      numeric -- treating it as numeric would silently impose a fake
      ordering on objects.
    - One object is used as the reference category (statsmodels default,
      first alphabetically) -- documented here so it's not a silent surprise
      when reading the coefficient table.
    - High Gap-occlusion correlation: VIF is computed and reported honestly,
      not hidden, per the pre-registered pass criterion.
    - Per-object logistic fits will fail to converge or be underpowered for
      objects with too few trials or too little failure-label variance --
      this is caught and reported as "insufficient data" for that object,
      not silently skipped without a trace.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import shap


N_BOOTSTRAP = 1000
RNG_SEED = 42


def fit_models(df: pd.DataFrame):
    """Fit Models A, B, C. Returns dict of fitted statsmodels results."""
    model_a = smf.logit("failure ~ true_occlusion + C(object_id)", data=df).fit(disp=0)
    model_b = smf.logit("failure ~ true_occlusion + Gap_initial + C(object_id)", data=df).fit(disp=0)
    model_c = smf.logit("failure ~ true_occlusion + Gap_max + C(object_id)", data=df).fit(disp=0)
    return {"A": model_a, "B": model_b, "C": model_c}


def compute_vif(df: pd.DataFrame, gap_col: str = "Gap_initial") -> float:
    """VIF of the Gap column against true_occlusion. Pre-registered flag: >10."""
    X = df[["true_occlusion", gap_col]].copy()
    X = sm.add_constant(X)
    vif = variance_inflation_factor(X.values, X.columns.get_loc(gap_col))
    return float(vif)


def bootstrap_auc_delta(df: pd.DataFrame, formula_a: str, formula_b: str,
                         n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED):
    """Bootstrap CI on delta-AUC (formula_b's model vs formula_a's model).
    Refits both models on each resample -- slow but correct; this is a
    one-off analysis step, not something run per-trial."""
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = df.iloc[idx].reset_index(drop=True)
        # skip resamples with degenerate outcome (all 0s or all 1s) or a
        # missing object category -- can't fit logit on those
        if sample["failure"].nunique() < 2:
            continue
        try:
            ma = smf.logit(formula_a, data=sample).fit(disp=0)
            mb = smf.logit(formula_b, data=sample).fit(disp=0)
            auc_a = roc_auc_score(sample["failure"], ma.predict(sample))
            auc_b = roc_auc_score(sample["failure"], mb.predict(sample))
            deltas.append(auc_b - auc_a)
        except Exception:
            continue  # non-convergence on a resample is expected occasionally; skip it
    deltas = np.array(deltas)
    if len(deltas) < n_boot * 0.5:
        print(f"  WARNING: only {len(deltas)}/{n_boot} bootstrap resamples converged. "
              f"CI below is based on fewer resamples than requested.")
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(ci_low), float(ci_high), len(deltas)


def rf_shap_check(df: pd.DataFrame, feature_cols=None):
    """Random Forest + SHAP cross-check: does Gap rank in the top-2 features?
    object_id is one-hot encoded as a categorical feature, per PREREG.md."""
    feature_cols = feature_cols or ["true_occlusion", "Gap_initial", "Gap_max", "clutter_count"]
    X = df[feature_cols].copy()
    X = pd.concat([X, pd.get_dummies(df["object_id"], prefix="obj")], axis=1)
    y = df["failure"]

    rf = RandomForestClassifier(max_depth=4, n_estimators=300, random_state=RNG_SEED)
    rf.fit(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    cv_auc = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    # SHAP's output shape varies by version/estimator:
    #   - list of per-class arrays (older sklearn/shap combos)
    #   - 3D array (n_samples, n_features, n_classes) (newer shap)
    #   - 2D array (n_samples, n_features) (regressors, or binary already collapsed)
    # Handle all three rather than assuming one.
    if isinstance(shap_values, list):
        sv = shap_values[1]  # positive class
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]  # positive class is index 1 for binary classification
    else:
        sv = shap_values
    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)

    return {
        "rf_cv_auc_mean": float(cv_auc.mean()),
        "rf_cv_auc_std": float(cv_auc.std()),
        "shap_importance_ranked": importance,
        "top2_features": list(importance.index[:2]),
    }


def per_object_descriptive_table(df: pd.DataFrame, gap_col: str = "Gap_initial"):
    """Per-object logistic fit (occlusion + Gap only, no fixed effect needed
    since each fit is already restricted to one object). Reports coefficient
    sign/direction per object -- this is the generalization evidence per
    PREREG.md, not per-object significance (expected underpowered at N=30-50)."""
    rows = []
    for obj, sub in df.groupby("object_id"):
        n = len(sub)
        n_fail = sub["failure"].sum()
        if sub["failure"].nunique() < 2 or n < 10:
            rows.append({
                "object_id": obj, "n_trials": n, "n_failures": int(n_fail),
                "gap_coef": None, "p_value": None, "note": "insufficient data to fit"
            })
            continue
        try:
            m = smf.logit(f"failure ~ true_occlusion + {gap_col}", data=sub).fit(disp=0)
            rows.append({
                "object_id": obj,
                "n_trials": n,
                "n_failures": int(n_fail),
                "gap_coef": float(m.params[gap_col]),
                "p_value": float(m.pvalues[gap_col]),
                "note": "sig at p<0.05" if m.pvalues[gap_col] < 0.05 else "not sig (expected at this N)",
            })
        except Exception as e:
            rows.append({
                "object_id": obj, "n_trials": n, "n_failures": int(n_fail),
                "gap_coef": None, "p_value": None, "note": f"fit failed: {e}"
            })
    return pd.DataFrame(rows)


def run_full_c1_package(df: pd.DataFrame):
    print("=" * 70)
    print("C1: Calibration Gap as an Independent Failure Predictor")
    print("=" * 70)

    models = fit_models(df)
    print("\n--- Model B (occlusion + Gap_initial + object fixed effect) ---")
    print(models["B"].summary().tables[1])

    gap_coef = models["B"].params["Gap_initial"]
    gap_p = models["B"].pvalues["Gap_initial"]
    odds_ratio = np.exp(gap_coef)
    or_ci = np.exp(models["B"].conf_int().loc["Gap_initial"])
    print(f"\nGap_initial coefficient: {gap_coef:.4f} (p={gap_p:.4g})")
    print(f"Odds ratio: {odds_ratio:.3f}  95% CI: [{or_ci[0]:.3f}, {or_ci[1]:.3f}]")

    vif = compute_vif(df, "Gap_initial")
    print(f"\nVIF(Gap_initial, true_occlusion) = {vif:.2f}"
          + ("  <-- FLAG: >10, weakens independence claim" if vif > 10 else "  (OK, <=10)"))

    print("\nBootstrapping delta-AUC (Model B vs Model A)... this takes a moment.")
    mean_d, lo, hi, n_ok = bootstrap_auc_delta(
        df,
        "failure ~ true_occlusion + C(object_id)",
        "failure ~ true_occlusion + Gap_initial + C(object_id)",
        n_boot=200,  # reduced for interactive runs; use N_BOOTSTRAP=1000 for final numbers
    )
    print(f"Delta-AUC: mean={mean_d:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  ({n_ok} resamples used)")
    print("  Pass criterion (CI excludes 0):", "PASS" if lo > 0 else "FAIL / INCONCLUSIVE")

    print("\nRandom Forest + SHAP cross-check...")
    rf_result = rf_shap_check(df)
    print(f"RF 5-fold CV AUC: {rf_result['rf_cv_auc_mean']:.3f} +/- {rf_result['rf_cv_auc_std']:.3f}")
    print("SHAP importance ranking:")
    print(rf_result["shap_importance_ranked"])
    gap_in_top2 = any("Gap" in f for f in rf_result["top2_features"])
    print(f"Gap in top-2 SHAP features: {'PASS' if gap_in_top2 else 'FAIL'} -> {rf_result['top2_features']}")

    print("\nPer-object descriptive table (generalization evidence):")
    per_obj = per_object_descriptive_table(df)
    print(per_obj.to_string(index=False))
    signs = per_obj["gap_coef"].dropna()
    consistent = (signs > 0).all() or (signs < 0).all()
    print(f"\nCoefficient sign consistent across all objects with usable data: "
          f"{'PASS' if consistent else 'FAIL -- sign flips across objects, investigate'}")

    return {
        "models": models,
        "vif": vif,
        "bootstrap_auc_delta": (mean_d, lo, hi, n_ok),
        "rf_shap": rf_result,
        "per_object_table": per_obj,
    }


if __name__ == "__main__":
    import os
    mock_path = os.path.join(os.path.dirname(__file__), "..", "mock-pipeline", "data", "mock_trials.csv")
    if not os.path.exists(mock_path):
        raise FileNotFoundError(
            f"No mock data at {mock_path}. Run mock_data_generator.py first:\n"
            f"  python3 mock-pipeline/mock_data_generator.py"
        )
    df = pd.read_csv(mock_path)
    results = run_full_c1_package(df)