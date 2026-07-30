"""
power_analysis_c3.py

Monte Carlo power analysis for C3's three-way interaction test
(failure ~ occlusion * clutter_count * Gap_high). Closed-form power for
interaction terms in logistic regression isn't reliable -- this simulates
data under a specified interaction effect size across candidate sample
sizes and reports the empirical fraction of runs where the interaction
term is detected at p < 0.05, so the experiment grid size (n_seeds in
run_trial.py's build_experiment_grid) can be chosen against an actual
power target instead of guessing.

Usage:
    python power_analysis_c3.py --effect-log-odds 1.0 --target-power 0.8

Once you have a real pilot batch (even ~100 trials), re-run this with
--effect-log-odds set to the actual fitted interaction coefficient from
analysis_c3.py -- the default here is a guess, not measured.
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OCCLUSION_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
CLUTTER_LEVELS = [0, 1, 3, 5]
CAMERA_VARIANTS = [0, 1]
LIGHT_LEVELS = [0.5, 1.0]
CELLS_PER_SEED = len(OCCLUSION_LEVELS) * len(CLUTTER_LEVELS) * len(CAMERA_VARIANTS) * len(LIGHT_LEVELS)


def simulate_dataset(n_seeds: int, interaction_log_odds: float, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _ in range(n_seeds):
        for occ in OCCLUSION_LEVELS:
            for clutter in CLUTTER_LEVELS:
                for _cam in CAMERA_VARIANTS:
                    for _light in LIGHT_LEVELS:
                        gap_high = rng.integers(0, 2)
                        logit_p = (
                            -2.0
                            + 2.0 * occ
                            + 0.15 * clutter
                            + 0.5 * gap_high
                            + interaction_log_odds * occ * clutter * gap_high
                        )
                        p_fail = 1.0 / (1.0 + np.exp(-logit_p))
                        failure = int(rng.random() < p_fail)
                        rows.append(dict(occlusion=occ, clutter_count=clutter,
                                          Gap_high=gap_high, failure=failure))
    return pd.DataFrame(rows)


def empirical_power(n_seeds: int, interaction_log_odds: float, n_reps: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_reps):
        df = simulate_dataset(n_seeds, interaction_log_odds, rng)
        try:
            model = smf.logit(
                "failure ~ occlusion * clutter_count * Gap_high", data=df
            ).fit(disp=0)
            pval = model.pvalues.get("occlusion:clutter_count:Gap_high", np.nan)
            if not np.isnan(pval) and pval < 0.05:
                hits += 1
        except Exception:
            continue
    return hits / n_reps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--effect-log-odds", type=float, default=1.0)
    p.add_argument("--target-power", type=float, default=0.8)
    p.add_argument("--n-reps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"Cells per seed (occlusion x clutter x camera x light): {CELLS_PER_SEED}")
    print(f"Detecting interaction effect (log-odds) = {args.effect_log_odds}\n")
    print(f"{'n_seeds':>8} {'N_trials':>10} {'power':>8}")

    for n_seeds in [1, 2, 3, 4, 5, 6, 8, 10]:
        power = empirical_power(n_seeds, args.effect_log_odds, args.n_reps, args.seed)
        n_trials = n_seeds * CELLS_PER_SEED
        flag = "  <- meets target" if power >= args.target_power else ""
        print(f"{n_seeds:>8} {n_trials:>10} {power:>8.2f}{flag}")

    print("\nThis assumes your real Gap-failure relationship is at least as strong as")
    print("--effect-log-odds. Re-run with the actual fitted coefficient from your first")
    print("~100-trial pilot batch (analysis_c3.py's interaction term) once you have it.")


if __name__ == "__main__":
    main()