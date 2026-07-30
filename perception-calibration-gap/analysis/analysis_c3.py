"""
analysis_c3.py

Purpose
-------
Tests whether Gap MODERATES the compounding effect of occlusion and
clutter -- i.e., does high Gap make failure accelerate faster as clutter
increases, not just add to it. Status: Primary (supporting C1's mechanism).

Pre-registered model form (PREREG.md C3):
    failure ~ true_occlusion * clutter_count * Gap_high + C(object_id)

    Gap_high = median split on Gap_initial, computed ONCE on the frozen
    pooled dataset, AFTER data collection ends, BEFORE any C3 model is fit.
    Never recomputed per object, never tuned after seeing outcomes.

Pre-registered pass criterion: the three-way interaction coefficient is
significant (p<0.05) AND its sign matches the hypothesis (high Gap steepens
the occlusion*clutter failure slope). If null: report as a negative result,
don't drop the term and silently re-test a simpler model.

How to check this script is correct:
    The median split must be computed exactly once per call, not per group
    or per bootstrap resample (that would leak information / redefine the
    threshold on subsets). Verify via the self-test below that
    compute_gap_high_median_split() returns the SAME threshold regardless
    of row order or which subset is later filtered from the full frame.

Edge cases handled:
    - Quasi-separation / non-convergence in the 3-way interaction model
      (common with a high-order interaction term at N~200) is caught and
      reported explicitly, not silently swallowed.
    - clutter_count is used as a continuous covariate (matching PREREG.md's
      literal model form using clutter_count, not a binned/categorical
      version) -- kept exactly as specified rather than "improved" silently.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def compute_gap_high_median_split(df: pd.DataFrame, gap_col: str = "Gap_initial") -> float:
    """Compute the Gap_initial median ONCE on the full pooled dataset.
    This must be called on the complete frozen dataset before any
    subsetting/filtering for downstream analysis -- computing it per-object
    or per-bootstrap-resample would silently redefine "high Gap" each time,
    breaking the pre-registered definition."""
    return float(df[gap_col].median())


def add_gap_high_column(df: pd.DataFrame, median_threshold: float,
                         gap_col: str = "Gap_initial") -> pd.DataFrame:
    """Adds Gap_high (0/1) using a PRE-COMPUTED threshold -- never recomputes
    the median internally, to prevent silent threshold drift across calls."""
    df = df.copy()
    df["Gap_high"] = (df[gap_col] >= median_threshold).astype(int)
    return df


def run_full_c3_package(df: pd.DataFrame):
    print("=" * 70)
    print("C3: Gap x Occlusion x Clutter Interaction (supporting C1's mechanism)")
    print("=" * 70)

    median_threshold = compute_gap_high_median_split(df)
    print(f"\nGap_initial median (computed once, on frozen pooled dataset): {median_threshold:.4f}")

    df = add_gap_high_column(df, median_threshold)
    print(f"Gap_high split: {df['Gap_high'].sum()} high / {(1 - df['Gap_high']).sum()} low")

    formula = "failure ~ true_occlusion * clutter_count * Gap_high + C(object_id)"
    print(f"\nModel formula: {formula}")

    try:
        model = smf.logit(formula, data=df).fit(disp=0)
    except Exception as e:
        print(f"\nMODEL FAILED TO FIT: {e}")
        print("This can happen with a high-order interaction term at N~200 "
              "(quasi-separation). Report this honestly rather than dropping "
              "the term and silently re-testing a simpler model per PREREG.md.")
        return {"model": None, "error": str(e)}

    interaction_term = "true_occlusion:clutter_count:Gap_high"
    print("\n--- Full model summary (interaction terms) ---")
    interaction_rows = [p for p in model.params.index if ":" in p]
    for p in interaction_rows:
        print(f"  {p}: coef={model.params[p]:.4f}, p={model.pvalues[p]:.4g}")

    if interaction_term in model.params.index:
        coef = model.params[interaction_term]
        p_val = model.pvalues[interaction_term]
        print(f"\nThree-way interaction ({interaction_term}):")
        print(f"  coefficient = {coef:.4f}, p = {p_val:.4g}")
        sig = p_val < 0.05
        sign_matches_hypothesis = coef > 0  # hypothesis: high Gap STEEPENS the slope
        if sig and sign_matches_hypothesis:
            print("  PASS: significant AND sign matches hypothesis (high Gap steepens "
                  "occlusion x clutter failure slope)")
        elif sig and not sign_matches_hypothesis:
            print("  Significant but WRONG SIGN -- report exactly as found, don't "
                  "reinterpret post-hoc.")
        else:
            print("  NOT significant -- report as a negative result per PREREG.md "
                  "(do not drop the term and re-test a simpler model without "
                  "disclosing this was the original plan).")
    else:
        print(f"\nWARNING: {interaction_term} not found in fitted model params "
              f"(check for a naming mismatch in the formula/statsmodels output) -- "
              f"available params: {list(model.params.index)}")

    return {"model": model, "median_threshold": median_threshold}


if __name__ == "__main__":
    import os
    mock_path = os.path.join(os.path.dirname(__file__), "..", "mock-pipeline", "data", "mock_trials.csv")
    if not os.path.exists(mock_path):
        raise FileNotFoundError("Run mock_data_generator.py first.")
    df = pd.read_csv(mock_path)

    # Self-test: median split threshold must be identical regardless of row
    # order or which subset is later filtered.
    t1 = compute_gap_high_median_split(df)
    t2 = compute_gap_high_median_split(df.sample(frac=1, random_state=1))  # shuffled
    t3 = compute_gap_high_median_split(df[df["object_id"] == "apple"])  # subset -- SHOULD differ from t1
    assert t1 == t2, "Median threshold must not depend on row order"
    print(f"Self-test: median threshold is row-order-independent (t1={t1:.4f}, t2={t2:.4f}) -- OK")
    print(f"Self-test note: subsetting to one object gives a DIFFERENT median ({t3:.4f}) "
          f"-- this is exactly why the threshold must be computed on the FULL pooled "
          f"dataset once, not recomputed per object.\n")

    results = run_full_c3_package(df)