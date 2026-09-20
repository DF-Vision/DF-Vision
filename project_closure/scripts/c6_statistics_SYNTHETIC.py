"""
c6_statistics_SYNTHETIC.py -- SYNTHETIC - NOT A RESULT

C6: re-establishes the synthetic statistics claims at the task's stated
minimums (>=1000 null datasets, >=500 planted datasets per effect size,
power at 3 N's, >=100 repeated conformal splits). Extends (does not
duplicate) project_overnight/scripts/t7_synthetic_recovery_check.py and
perception-calibration-gap/analysis/power_analysis_c1.py, which used far
smaller repeat counts. See project_closure/CRITERIA.md's C6 section.

Nothing here is real trial data. Nothing here validates failure
prediction on real hardware.
"""
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "mock-pipeline"))

from mock_data_generator import generate_mock_trials, TRUE_COEFS, DEFAULT_N_PER_OBJECT
from analysis_c1 import fit_models
from conformal_gate import ConformalGate, three_split
from power_analysis_c1 import empirical_power, scaled_n_per_object

RANDOM_STATE = 20260921
TRUE_B_GAP = TRUE_COEFS["b_gap"]

OUT_PATH = Path(__file__).parent / "c6_statistics_SYNTHETIC_output.txt"
_lines = []


def log(s=""):
    print(s)
    _lines.append(s)


# --- (a) Null false-positive rate, >=1000 datasets, exact binomial CI ---

def null_false_positive_rate(n_repeats=1000, seed=RANDOM_STATE):
    log(f"\n{'='*78}\n(a) Null false-positive rate, n_repeats={n_repeats}, N={sum(DEFAULT_N_PER_OBJECT.values())}\n{'='*78}")
    n_significant, n_singular, n_fit_ok = 0, 0, 0
    for i in range(n_repeats):
        df = generate_mock_trials(seed=seed + i, gap_signal_multiplier=0.0)
        try:
            model_b = fit_models(df)["B"]
        except np.linalg.LinAlgError:
            n_singular += 1
            continue
        n_fit_ok += 1
        if model_b.pvalues["Gap_initial"] < 0.05:
            n_significant += 1

    fpr = n_significant / n_fit_ok if n_fit_ok else float("nan")
    ci = binomtest(n_significant, n_fit_ok, p=0.05).proportion_ci(confidence_level=0.95) if n_fit_ok else None
    log(f"Fit succeeded: {n_fit_ok}/{n_repeats} (singular failures: {n_singular})")
    log(f"False-positive rate (p<0.05 on null data): {n_significant}/{n_fit_ok} = {fpr:.4f}")
    if ci:
        log(f"Exact binomial 95% CI: [{ci.low:.4f}, {ci.high:.4f}]")
        nominal_in_ci = ci.low <= 0.05 <= ci.high
        log(f"Nominal 0.05 within CI: {'YES -- well-calibrated' if nominal_in_ci else 'NO -- miscalibrated'}")
    return fpr, ci, n_fit_ok, n_singular


def variance_equality_check(seed=RANDOM_STATE):
    """Regression test (also run as a pytest assertion in
    tests/test_c6_statistics.py): Gap's variance must be IDENTICAL
    between the null and planted generators at the same seed -- this is
    the exact property whose breakage caused T7's prior-session bug."""
    log(f"\n{'='*78}\nGap-variance equality check (null vs planted generator, same seed)\n{'='*78}")
    df_null = generate_mock_trials(seed=seed, gap_signal_multiplier=0.0)
    df_planted = generate_mock_trials(seed=seed, gap_signal_multiplier=1.0)
    var_null = df_null["Gap_initial"].var()
    var_planted = df_planted["Gap_initial"].var()
    log(f"Gap_initial variance -- null: {var_null:.8f}, planted: {var_planted:.8f}")
    identical = np.isclose(var_null, var_planted, rtol=0, atol=1e-12)
    log(f"Identical: {'YES' if identical else 'NO -- regression of the T7 null-generator bug'}")
    return identical, var_null, var_planted


# --- (b) Planted effect recovery, >=500 per effect size, 3 effect sizes ---

def planted_recovery_by_effect_size(n_repeats=500, multipliers=(0.5, 1.0, 2.0), seed=RANDOM_STATE + 2000):
    log(f"\n{'='*78}\n(b) Planted effect recovery, n_repeats={n_repeats} per effect size\n{'='*78}")
    results = {}
    for mult in multipliers:
        true_coef = TRUE_B_GAP * mult
        correct_sign, in_ci = 0, 0
        coefs = []
        n_ok = 0
        for i in range(n_repeats):
            df = generate_mock_trials(seed=seed + hash((mult, i)) % 1_000_000, gap_signal_multiplier=mult)
            try:
                m = fit_models(df)["B"]
            except np.linalg.LinAlgError:
                continue
            n_ok += 1
            coef = m.params["Gap_initial"]
            coefs.append(coef)
            if coef > 0:
                correct_sign += 1
            ci_lo, ci_hi = m.conf_int().loc["Gap_initial"]
            if ci_lo <= true_coef <= ci_hi:
                in_ci += 1

        coefs = np.array(coefs)
        bias = float(coefs.mean() - true_coef) if n_ok else float("nan")
        sign_rate = correct_sign / n_ok if n_ok else float("nan")
        coverage = in_ci / n_ok if n_ok else float("nan")
        log(f"\nmultiplier={mult} (true b_gap={true_coef:.3f}): n_ok={n_ok}/{n_repeats}")
        log(f"  Sign-correct rate: {sign_rate:.4f}")
        log(f"  Mean recovered coef: {coefs.mean():.4f}  Bias (mean - true): {bias:+.4f}")
        log(f"  95% CI coverage (fraction of CIs containing true coef): {coverage:.4f} (nominal target: 0.95)")
        results[mult] = {"n_ok": n_ok, "sign_rate": sign_rate, "bias": bias, "ci_coverage": coverage,
                          "mean_coef": float(coefs.mean())}
    return results


# --- (c) Power curve at 3 N's + minimal detectable effect + conformal spread ---

def power_curve(seed=RANDOM_STATE + 3000, n_reps=300):
    log(f"\n{'='*78}\n(c) Power curve at N=90 (planned), N=120, N=180\n{'='*78}")
    scale_factors = {90: 1.0, 120: 120 / 90, 180: 180 / 90}
    results = {}
    for target_n, sf in scale_factors.items():
        r = empirical_power(sf, gap_signal_multiplier=1.0, n_reps=n_reps, base_seed=seed)
        n_objects = len(r["n_per_object"])
        per_object_n = target_n / n_objects
        log(f"\nN~{target_n} (scale_factor={sf:.3f}, actual N_total={r['n_total']}, "
            f"{n_objects} objects, ~{per_object_n:.0f}/object):")
        log(f"  Power (p<0.05, correct sign) at nominal effect: {r['power']:.3f}")
        log(f"  {'UNDERPOWERED (<0.8)' if r['power'] < 0.8 else 'Meets conventional 0.8 power target'}")
        log(f"  Stable for per-object conformal calibration (>=20 successful/object, "
            f"conformal_gate.py's own threshold)? "
            f"{'Marginal/NO' if per_object_n * 0.5 < 20 else 'Plausible'} "
            f"(~{per_object_n * 0.5:.0f} successful trials/object at ~50% baseline success rate, illustrative)")
        results[target_n] = r

    # Minimal detectable effect at N=90: coarse search over multipliers.
    log(f"\nMinimal detectable effect (multiplier clearing 0.8 power) at N=90, n_reps={n_reps}:")
    mde = None
    for mult in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        r = empirical_power(1.0, gap_signal_multiplier=mult, n_reps=n_reps, base_seed=seed + 1)
        log(f"  multiplier={mult}: power={r['power']:.3f}")
        if r["power"] >= 0.8 and mde is None:
            mde = mult
    log(f"  Approximate MDE (smallest multiplier tested clearing 0.8 power): "
        f"{mde if mde is not None else 'NONE of the tested multipliers reached 0.8 power'}")

    log(f"\nSanity/extent check (not a formal C6(c) requirement, but needed to state the "
        f"conclusion honestly): how much LARGER would N need to be, at the nominal "
        f"effect size, to approach 0.8 power? Sweeping scale_factor far beyond the "
        f"180 the task asks about:")
    for sf in [4.0, 8.0, 16.0]:
        r = empirical_power(sf, gap_signal_multiplier=1.0, n_reps=n_reps, base_seed=seed + 2)
        log(f"  N={r['n_total']}: power={r['power']:.3f}")
    log(f"  (If even N=1440 -- 16x the planned N -- does not clear 0.8 power at the "
        f"nominal effect size, this is strong evidence the study, as currently "
        f"designed around Model B's specific significance test, cannot reach "
        f"conventional power at any N remotely achievable by real data collection.)")
    return results, mde


def conformal_coverage_spread(n_splits=100, alpha=0.10, seed=RANDOM_STATE + 4000):
    log(f"\n{'='*78}\n(c, continued) Conformal coverage distribution over {n_splits} repeated splits, N=90\n{'='*78}")
    coverages, flag_rates = [], []
    for i in range(n_splits):
        df = generate_mock_trials(seed=seed + i, gap_signal_multiplier=1.0)
        calib_df, test_df = three_split(df, calib_frac=0.7, seed=seed + i)
        gate = ConformalGate(alpha=alpha)
        gate.fit(calib_df)
        cov = gate.empirical_coverage(test_df)
        coverages.append(cov["pooled_coverage"])
        flag_rates.append(cov["pooled_flag_rate"])
    coverages = np.array(coverages)
    target = 1 - alpha
    log(f"Coverage over {n_splits} splits: mean={coverages.mean():.4f}, std={coverages.std():.4f}, "
        f"min={coverages.min():.4f}, max={coverages.max():.4f} (target={target:.3f})")
    within_5pp_frac = float(np.mean(np.abs(coverages - target) <= 0.05))
    log(f"Fraction of splits within +-5pp of target: {within_5pp_frac:.3f}")
    log(f"(Motivated by the prior session's single-seed 0.955-vs-0.900 finding, flagged but not "
        f"resolved as noise-vs-systematic -- this IS that resolution: {n_splits} splits show the "
        f"spread directly rather than one seed's point estimate.)")
    return coverages, within_5pp_frac


def main():
    _lines.append("SYNTHETIC - NOT A RESULT")
    _lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    _lines.append(f"RANDOM_STATE base = {RANDOM_STATE}")
    t0 = time.time()

    null_false_positive_rate(n_repeats=1000)
    variance_equality_check()
    planted_recovery_by_effect_size(n_repeats=500)
    power_curve(n_reps=300)
    conformal_coverage_spread(n_splits=100)

    log(f"\n{'='*78}\nTotal wall-clock: {time.time()-t0:.1f}s. SYNTHETIC. NOT A RESULT.\n{'='*78}")
    OUT_PATH.write_text("\n".join(_lines) + "\n")
    print(f"\nWritten: {OUT_PATH}")


if __name__ == "__main__":
    main()
