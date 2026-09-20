"""
tests/test_c6_statistics.py

C6: fast pytest-level regression tests for the properties
c6_statistics_SYNTHETIC.py verifies at scale (>=1000/>=500 repeats --
too slow for a routine test run). This file uses small repeat counts
sufficient to test CORRECTNESS of the computation (exact binomial CI
formula, variance-equality property, coverage-fraction arithmetic), not
to re-derive the full-scale statistical conclusions themselves (those
live in project_closure/scripts/c6_statistics_SYNTHETIC_output.txt,
regenerable via that script).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "mock-pipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "project_closure" / "scripts"))

from mock_data_generator import generate_mock_trials, DEFAULT_N_PER_OBJECT
import c6_statistics_SYNTHETIC as c6


def test_default_n_per_object_sums_within_75_90_range():
    """C6(a)'s N requirement -- confirms the generator's default is
    actually in range, not asserted blindly."""
    total = sum(DEFAULT_N_PER_OBJECT.values())
    assert 75 <= total <= 90, f"DEFAULT_N_PER_OBJECT sums to {total}, expected [75,90]"


def test_gap_variance_identical_between_null_and_planted_generator():
    """Regression test for T7's prior-session bug (project_overnight/LOG.md):
    gap_signal_multiplier=0.0 must zero Gap's EFFECT on failure, not Gap's
    own variance. This is the exact property that was broken and fixed."""
    identical, var_null, var_planted = c6.variance_equality_check(seed=42)
    assert identical, f"Gap variance differs between null ({var_null}) and planted ({var_planted}) generators"
    assert var_null > 0, "Gap variance must be nonzero in the first place for this check to be meaningful"


def test_gap_variance_identical_across_several_seeds():
    """Not just one seed -- confirms the property holds generally."""
    for seed in [1, 2, 3, 100, 999]:
        identical, var_null, var_planted = c6.variance_equality_check(seed=seed)
        assert identical, f"seed={seed}: variance differs (null={var_null}, planted={var_planted})"


def test_exact_binomial_ci_matches_scipy_directly():
    """Confirms null_false_positive_rate() uses scipy's EXACT binomial CI
    (not a normal approximation) by cross-checking against a direct
    scipy.stats.binomtest call on the same counts."""
    fpr, ci, n_fit_ok, n_singular = c6.null_false_positive_rate(n_repeats=50, seed=5000)
    n_significant = round(fpr * n_fit_ok)
    expected_ci = binomtest(n_significant, n_fit_ok, p=0.05).proportion_ci(confidence_level=0.95)
    assert ci.low == pytest.approx(expected_ci.low)
    assert ci.high == pytest.approx(expected_ci.high)


def test_planted_recovery_reports_all_three_effect_sizes():
    results = c6.planted_recovery_by_effect_size(n_repeats=20, multipliers=(0.5, 1.0, 2.0), seed=6000)
    assert set(results.keys()) == {0.5, 1.0, 2.0}
    for mult, r in results.items():
        assert r["n_ok"] > 0
        assert 0.0 <= r["sign_rate"] <= 1.0
        assert 0.0 <= r["ci_coverage"] <= 1.0


def test_planted_recovery_bias_is_signed_correctly():
    """Bias = mean_recovered - true_coef; for a multiplier > 0 with a
    correctly-signed estimator, this should be a small number relative to
    the true coefficient itself, not wildly off (sanity bound, not a tight
    numerical claim)."""
    results = c6.planted_recovery_by_effect_size(n_repeats=30, multipliers=(1.0,), seed=7000)
    r = results[1.0]
    true_coef = c6.TRUE_B_GAP * 1.0
    assert abs(r["bias"]) < true_coef, (
        f"bias ({r['bias']}) is larger in magnitude than the true coefficient itself "
        f"({true_coef}) -- likely a sign/formula error, not just noise"
    )


def test_conformal_coverage_spread_reports_min_max_std_not_just_mean():
    coverages, within_5pp_frac = c6.conformal_coverage_spread(n_splits=15, seed=8000)
    assert len(coverages) == 15
    assert coverages.std() >= 0.0
    assert 0.0 <= within_5pp_frac <= 1.0
    # the whole point of this check is that spread is reported, not collapsed to one number
    assert coverages.max() >= coverages.min()


def test_power_curve_runs_at_all_three_planned_ns():
    results, mde = c6.power_curve(n_reps=15)
    assert set(results.keys()) == {90, 120, 180}
    for target_n, r in results.items():
        assert r["n_total"] > 0
        assert 0.0 <= r["power"] <= 1.0
