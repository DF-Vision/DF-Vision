#!/usr/bin/env python3
"""
project_overnight/scripts/t7_synthetic_recovery_check.py

T7: verifies the EXISTING Gap-only statistics package
(analysis/analysis_c1.py's fit_models, mock-pipeline/mock_data_generator.py's
generate_mock_trials, mock-pipeline/conformal_gate.py's ConformalGate)
against the pre-committed criteria in project_overnight/CRITERIA.md:
  - planted-effect recovery: correct Gap sign in >=90% of 50 seeded
    datasets, magnitude within 3x of the generator's own b_gap=2.5.
  - conformal coverage on planted data within +-5pp of target.
  - null-data false-positive rate across >=100 seeded null datasets,
    reported honestly (informational, not a hard gate per CRITERIA.md).

All outputs SYNTHETIC -- never citable as C1/C1-ext/C4/C4-ext evidence.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "mock-pipeline"))

import numpy as np

# WORKAROUND, this session: `shap` is broken in this environment (numba/
# coverage package version conflict -- AttributeError: module 'coverage'
# has no attribute 'types', confirmed in project_overnight/STATE.md's T0
# environment check). analysis_c1.py imports shap at MODULE LEVEL, so
# even fit_models() (which doesn't need shap) can't be imported without
# it. Stubbing sys.modules['shap'] with a dummy BEFORE importing
# analysis_c1 -- purely in-process, does not touch the installed
# environment -- lets everything except rf_shap_check() (not called by
# this script) work. rf_shap_check() itself remains genuinely broken and
# UNVERIFIED this session -- not silently worked around, see this
# script's own printed warning below.
import sys as _sys
import types as _types
if "shap" not in _sys.modules:
    _sys.modules["shap"] = _types.ModuleType("shap")
    print("[WORKAROUND] Stubbed sys.modules['shap'] (real shap import is broken in this "
          "environment) so analysis_c1.fit_models etc. can be imported. "
          "analysis_c1.rf_shap_check() itself is NOT called by this script and remains "
          "genuinely UNVERIFIED -- see project_overnight/STATE.md's environment check.")

from mock_data_generator import generate_mock_trials, TRUE_COEFS
from analysis_c1 import fit_models
from conformal_gate import ConformalGate, three_split

RANDOM_STATE = 20260920
N_REPEATS_PLANTED = 50
N_REPEATS_NULL = 100
TRUE_B_GAP = TRUE_COEFS["b_gap"]


def planted_recovery_check(n_repeats=N_REPEATS_PLANTED, seed=RANDOM_STATE):
    print(f"{'='*78}\nSYNTHETIC -- NOT A RESULT. Planted-effect recovery, n_repeats={n_repeats}\n{'='*78}")
    correct_sign = 0
    within_magnitude_band = 0
    coefs = []
    for i in range(n_repeats):
        df = generate_mock_trials(seed=seed + i, gap_signal_multiplier=1.0)
        models = fit_models(df); model_b = models["B"]
        gap_coef = model_b.params["Gap_initial"]
        coefs.append(gap_coef)
        if gap_coef > 0:  # generator's known planted direction (b_gap=2.5 > 0)
            correct_sign += 1
        if TRUE_B_GAP / 3.0 <= gap_coef <= TRUE_B_GAP * 3.0:
            within_magnitude_band += 1

    sign_rate = correct_sign / n_repeats
    magnitude_rate = within_magnitude_band / n_repeats
    print(f"Correct sign: {correct_sign}/{n_repeats} ({sign_rate:.1%}) "
          f"-- gate: >=90% -> {'PASS' if sign_rate >= 0.90 else 'FAIL'}")
    print(f"Within 3x magnitude band [{TRUE_B_GAP/3:.3f}, {TRUE_B_GAP*3:.3f}]: "
          f"{within_magnitude_band}/{n_repeats} ({magnitude_rate:.1%})")
    print(f"Mean recovered Gap_initial coefficient: {np.mean(coefs):.4f} "
          f"(generator's own b_gap={TRUE_B_GAP})")
    return sign_rate, magnitude_rate, coefs


def conformal_coverage_check(alpha=0.10, seed=RANDOM_STATE):
    print(f"\n{'='*78}\nSYNTHETIC -- NOT A RESULT. Conformal coverage on planted data, alpha={alpha}\n{'='*78}")
    df = generate_mock_trials(seed=seed, gap_signal_multiplier=1.0)
    calib_df, test_df = three_split(df, calib_frac=0.7, seed=seed)
    gate = ConformalGate(alpha=alpha)
    gate.fit(calib_df)
    coverage = gate.empirical_coverage(test_df)
    target = 1 - alpha
    within_5pp = abs(coverage["pooled_coverage"] - target) <= 0.05
    print(f"Pooled coverage: {coverage['pooled_coverage']:.3f} (target {target:.3f}) "
          f"-- gate: within +-5pp -> {'PASS' if within_5pp else 'FAIL'}")
    print(f"Flag rate: {coverage['pooled_flag_rate']:.3f}")
    return within_5pp, coverage


def null_false_positive_rate(n_repeats=N_REPEATS_NULL, seed=RANDOM_STATE + 1000):
    print(f"\n{'='*78}\nSYNTHETIC -- NOT A RESULT. Null-data false-positive rate, n_repeats={n_repeats}\n{'='*78}")
    n_significant = 0
    n_singular = 0
    n_fit_ok = 0
    for i in range(n_repeats):
        df = generate_mock_trials(seed=seed + i, gap_signal_multiplier=0.0)
        try:
            model_b = fit_models(df)["B"]
        except np.linalg.LinAlgError:
            # A singular Hessian on NULL data (no planted Gap-failure
            # relationship) is itself informative -- logged as its own
            # count, not silently skipped or treated as "not significant".
            n_singular += 1
            continue
        n_fit_ok += 1
        if model_b.pvalues["Gap_initial"] < 0.05:
            n_significant += 1
    fpr = n_significant / n_fit_ok if n_fit_ok > 0 else float("nan")
    well_calibrated = 0.025 <= fpr <= 0.10 if n_fit_ok > 0 else False
    print(f"Fit succeeded: {n_fit_ok}/{n_repeats} (singular-matrix failures: {n_singular}/{n_repeats} "
          f"-- reported, not hidden)")
    print(f"False-positive rate (p<0.05 on null data, of successful fits): "
          f"{n_significant}/{n_fit_ok} ({fpr:.1%})" if n_fit_ok else "N/A -- no successful fits")
    print(f"Well-calibrated (within 2x of nominal 0.05): {'YES' if well_calibrated else 'NO -- reported honestly, not a hard gate'}")
    return fpr, well_calibrated, n_singular


if __name__ == "__main__":
    planted_recovery_check()
    conformal_coverage_check()
    null_false_positive_rate()
    print(f"\n{'='*78}\nREMINDER: every number above is SYNTHETIC. Not citable as C1/C1-ext/C4/C4-ext evidence.\n{'='*78}")
