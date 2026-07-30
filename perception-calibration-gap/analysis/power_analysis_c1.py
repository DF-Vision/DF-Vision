"""
power_analysis_c1.py

Monte Carlo power analysis for C1's primary claim: the Gap_initial
coefficient in Model B (failure ~ true_occlusion + Gap_initial +
C(object_id)) is significant at p < 0.05, with odds-ratio CI excluding 1.

Unlike a generic simulator, this reuses mock_data_generator.py's ACTUAL
generative process (same detector-noise scales, same TRUE_COEFS, same
per-object structure) so the power estimate is grounded in the same
assumptions already encoded in the mock pipeline -- not a fresh guess.

Two things are varied independently:
    1. gap_signal_multiplier -- how strong the real Gap->failure effect is
       (1.0 = the "realistic-magnitude" default already used elsewhere in
       the project; >1.0 = optimistic scenarios in case the real effect
       turns out stronger than the conservative default).
    2. scale_factor -- how many trials per object, scaled proportionally
       off DEFAULT_N_PER_OBJECT (which sums to 195, close to the
       pre-registered 180-240 target).

For each (multiplier, scale_factor) cell, refit Model B on n_reps
independent simulated datasets and report the empirical fraction where
Gap_initial's p-value < 0.05 (the pre-registered criterion #1). This
directly reuses PREREG.md's Model B formula, not a simplified stand-in.

Usage:
    python power_analysis_c1.py --n-reps 300 --target-power 0.8
"""

import argparse
import os
import sys

import numpy as np
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mock-pipeline"))
from mock_data_generator import generate_mock_trials, DEFAULT_N_PER_OBJECT  # noqa: E402

MODEL_B_FORMULA = "failure ~ true_occlusion + Gap_initial + C(object_id)"


def scaled_n_per_object(scale_factor: float) -> dict:
    """Scale DEFAULT_N_PER_OBJECT proportionally, preserving the
    must-have/nice-to-have ratio already decided in PREREG.md."""
    return {obj: max(5, round(n * scale_factor)) for obj, n in DEFAULT_N_PER_OBJECT.items()}


def empirical_power(scale_factor: float, gap_signal_multiplier: float,
                     n_reps: int, base_seed: int) -> dict:
    n_per_object = scaled_n_per_object(scale_factor)
    hits = 0
    converged = 0
    coefs = []
    for rep in range(n_reps):
        seed = base_seed * 100_000 + rep
        df = generate_mock_trials(n_per_object=n_per_object, seed=seed,
                                   gap_signal_multiplier=gap_signal_multiplier)
        if df["failure"].nunique() < 2:
            continue
        try:
            m = smf.logit(MODEL_B_FORMULA, data=df).fit(disp=0, maxiter=100)
            pval = m.pvalues.get("Gap_initial", np.nan)
            coef = m.params.get("Gap_initial", np.nan)
            if np.isnan(pval):
                continue
            converged += 1
            coefs.append(coef)
            if pval < 0.05 and coef > 0:  # sign-aware: must be sig AND correct direction
                hits += 1
        except Exception:
            continue  # non-convergence / separation counts as a miss, not skipped silently
    n_total = sum(n_per_object.values())
    return {
        "n_total": n_total,
        "n_per_object": n_per_object,
        "power": hits / n_reps,
        "converged_fraction": converged / n_reps,
        "mean_recovered_coef": float(np.mean(coefs)) if coefs else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-reps", type=int, default=300)
    p.add_argument("--target-power", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    scale_factors_arg = os.environ.get("PA_SCALE_FACTORS")
    scale_factors = ([float(x) for x in scale_factors_arg.split(",")] if scale_factors_arg
                      else [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0])
    mult_arg = os.environ.get("PA_MULTIPLIERS")
    multipliers = ([float(x) for x in mult_arg.split(",")] if mult_arg
                   else [1.0, 2.0, 4.0])  # 1.0 = realistic default; >1 = optimistic scenarios

    print(f"True b_gap in generator = 2.5 (log-odds per unit Gap_initial), scaled by multiplier.")
    print(f"n_reps per cell = {args.n_reps}\n")

    for mult in multipliers:
        print("=" * 78)
        print(f"gap_signal_multiplier = {mult}  "
              f"({'REALISTIC default' if mult == 1.0 else 'optimistic scenario'})")
        print("=" * 78)
        print(f"{'scale':>6} {'N_total':>8} {'power':>7} {'converged%':>11} {'mean_coef':>10}")
        for sf in scale_factors:
            result = empirical_power(sf, mult, args.n_reps, args.seed)
            flag = "  <- meets target" if result["power"] >= args.target_power else ""
            print(f"{sf:>6.1f} {result['n_total']:>8} {result['power']:>7.2f} "
                  f"{result['converged_fraction']*100:>10.1f}% "
                  f"{result['mean_recovered_coef']:>10.3f}{flag}")
        print()


if __name__ == "__main__":
    main()