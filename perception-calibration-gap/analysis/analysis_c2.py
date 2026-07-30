"""
analysis_c2.py

Purpose
-------
SECONDARY/supporting contribution. Tests whether a learned recalibration
function (perceived_visibility -> true_visibility) recovers some of the
performance ceiling, and whether an MLP captures nonlinearity a stratified
linear map misses. Never let this read as central -- one figure, one table.

Pre-registered model forms (PREREG.md C2):
    - Stratified LinearRegression: one linear fit per occlusion bin
      (e.g. 3 bins: low/med/high true_occlusion)
    - MLPRegressor([perceived_visibility, Gap_initial, clutter_count] -> corrected_visibility)
    both fit on the same trial logs, compared via McNemar's test using
    Common Random Numbers (CRN) so the comparison is paired.

Pre-registered pass criterion: MLP beats stratified-linear on McNemar's
test AND both sit above the uncorrected baseline and below an oracle
(ground-truth-fed) success rate. If MLP does NOT beat linear: report
"miscalibration structure is largely linear within bins" -- still a valid,
reportable finding, not a failure.

How to check this script is correct:
    Run against mock data where the true relationship between
    perceived_visibility and true_visibility is close to linear (as
    constructed in mock_data_generator.py) -- the stratified-linear model
    should perform comparably to or better than the MLP here, since there's
    no real nonlinearity to capture. This is itself a useful check: if MLP
    "wins" by a lot on data we know is linear, something is wrong (probably
    overfitting) and needs investigating before trusting it on real data.

Edge cases handled:
    - Occlusion bins with too few samples to fit a stable linear model are
      reported explicitly, not silently merged into a neighboring bin.
    - McNemar's test requires binary correct/incorrect outcomes derived
      from the SAME held-out trials for both models (paired) -- this is
      enforced by evaluating both models on one shared test split, not two
      independent splits (which would break the pairing).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from statsmodels.stats.contingency_tables import mcnemar

RNG_SEED = 42
N_OCCLUSION_BINS = 3


def make_occlusion_bins(df: pd.DataFrame, n_bins: int = N_OCCLUSION_BINS) -> pd.Series:
    """Equal-width bins on true_occlusion (0-1), labeled 0..n_bins-1."""
    return pd.cut(df["true_occlusion"], bins=n_bins, labels=False, include_lowest=True)


def fit_stratified_linear(train_df: pd.DataFrame, n_bins: int = N_OCCLUSION_BINS):
    """One LinearRegression per occlusion bin: perceived_visibility -> true_visibility.
    Returns dict {bin_label: fitted_model}. Bins with <5 samples are skipped
    and reported, not silently merged."""
    bins = make_occlusion_bins(train_df, n_bins)
    models = {}
    skipped = []
    for b in sorted(bins.dropna().unique()):
        sub = train_df[bins == b]
        if len(sub) < 5:
            skipped.append((b, len(sub)))
            continue
        X = sub[["perceived_visibility_initial"]].values
        y = sub["true_visibility"].values
        m = LinearRegression().fit(X, y)
        models[b] = m
    return models, skipped


def predict_stratified_linear(test_df: pd.DataFrame, models: dict, n_bins: int = N_OCCLUSION_BINS):
    """Predict corrected_visibility using the bin-appropriate linear model.
    Rows whose bin has no fitted model fall back to identity (uncorrected)
    perceived_visibility -- reported, not silently substituted without a trace."""
    bins = make_occlusion_bins(test_df, n_bins)
    preds = np.zeros(len(test_df))
    fallback_count = 0
    for i, (idx, row) in enumerate(test_df.iterrows()):
        b = bins.loc[idx]
        if pd.isna(b) or b not in models:
            preds[i] = row["perceived_visibility_initial"]
            fallback_count += 1
        else:
            preds[i] = models[b].predict([[row["perceived_visibility_initial"]]])[0]
    return preds, fallback_count


def fit_mlp(train_df: pd.DataFrame, seed: int = RNG_SEED):
    X = train_df[["perceived_visibility_initial", "Gap_initial", "clutter_count"]].values
    y = train_df["true_visibility"].values
    mlp = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=seed)
    mlp.fit(X, y)
    return mlp


def run_full_c2_package(df: pd.DataFrame, seed: int = RNG_SEED):
    print("=" * 70)
    print("C2: Post-hoc Calibration Correction (SECONDARY, supporting)")
    print("=" * 70)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=seed)

    linear_models, skipped_bins = fit_stratified_linear(train_df)
    if skipped_bins:
        print(f"\nNote: {len(skipped_bins)} occlusion bin(s) skipped for stratified-linear "
              f"(too few samples): {skipped_bins}")

    linear_preds, fallback_n = predict_stratified_linear(test_df, linear_models)
    if fallback_n:
        print(f"Note: {fallback_n}/{len(test_df)} test rows fell back to uncorrected "
              f"perceived_visibility (no fitted model for their bin)")

    mlp = fit_mlp(train_df, seed=seed)
    mlp_preds = mlp.predict(
        test_df[["perceived_visibility_initial", "Gap_initial", "clutter_count"]].values
    )

    true_vis = test_df["true_visibility"].values
    uncorrected = test_df["perceived_visibility_initial"].values

    mae_uncorrected = np.mean(np.abs(uncorrected - true_vis))
    mae_linear = np.mean(np.abs(linear_preds - true_vis))
    mae_mlp = np.mean(np.abs(mlp_preds - true_vis))

    print(f"\nMAE (perceived_visibility vs true_visibility):")
    print(f"  Uncorrected (baseline): {mae_uncorrected:.4f}")
    print(f"  Stratified-linear corrected: {mae_linear:.4f}")
    print(f"  MLP corrected: {mae_mlp:.4f}")

    # McNemar's test via CRN: define "correct" as within a tolerance band
    # of true_visibility, using the SAME held-out test rows for both models
    # (paired by construction, since both draw from test_df).
    tolerance = 0.1
    linear_correct = (np.abs(linear_preds - true_vis) < tolerance).astype(int)
    mlp_correct = (np.abs(mlp_preds - true_vis) < tolerance).astype(int)

    # 2x2 contingency table: [both correct, linear only], [mlp only, both wrong]
    both_correct = int(((linear_correct == 1) & (mlp_correct == 1)).sum())
    linear_only = int(((linear_correct == 1) & (mlp_correct == 0)).sum())
    mlp_only = int(((linear_correct == 0) & (mlp_correct == 1)).sum())
    both_wrong = int(((linear_correct == 0) & (mlp_correct == 0)).sum())
    table = [[both_correct, linear_only], [mlp_only, both_wrong]]

    print(f"\nMcNemar contingency table (tolerance={tolerance}):")
    print(f"  both correct={both_correct}, linear-only-correct={linear_only}, "
          f"mlp-only-correct={mlp_only}, both wrong={both_wrong}")

    if linear_only + mlp_only == 0:
        print("  McNemar's test undefined: no discordant pairs (linear and MLP "
              "agree on every prediction's correctness) -- report as such, don't force a p-value.")
        mcnemar_result = None
    else:
        mcnemar_result = mcnemar(table, exact=(linear_only + mlp_only < 25))
        print(f"  McNemar statistic={mcnemar_result.statistic:.4f}, p={mcnemar_result.pvalue:.4g}")
        mlp_wins = mlp_only > linear_only and mcnemar_result.pvalue < 0.05
        if mcnemar_result.pvalue < 0.05:
            winner = "MLP" if mlp_only > linear_only else "stratified-linear"
            print(f"  PASS criterion check: {winner} significantly better (p<0.05)")
        else:
            print("  Not significant -- report as 'miscalibration structure is largely "
                  "linear within bins' (still a valid finding per PREREG.md)")

    # Pass criterion: both corrections should sit above uncorrected baseline MAE... i.e. lower MAE
    print(f"\nBoth corrections beat uncorrected baseline (lower MAE): "
          f"linear={'yes' if mae_linear < mae_uncorrected else 'NO'}, "
          f"mlp={'yes' if mae_mlp < mae_uncorrected else 'NO'}")

    return {
        "mae_uncorrected": mae_uncorrected, "mae_linear": mae_linear, "mae_mlp": mae_mlp,
        "mcnemar_result": mcnemar_result,
        "linear_models": linear_models, "mlp_model": mlp,
    }


if __name__ == "__main__":
    import os
    mock_path = os.path.join(os.path.dirname(__file__), "..", "mock-pipeline", "data", "mock_trials.csv")
    if not os.path.exists(mock_path):
        raise FileNotFoundError("Run mock_data_generator.py first.")
    df = pd.read_csv(mock_path)
    results = run_full_c2_package(df)