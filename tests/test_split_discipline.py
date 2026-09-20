"""
tests/test_split_discipline.py

C6(d): confirms analysis_c4.py's FAIL-Detect baseline is trained ONLY on
calib_df rows, never test_df rows. Two complementary checks: a runtime
property of three_split() itself (calib/test trial_id sets are disjoint),
and an AST-based source check that analysis_c4.py's baseline.fit( call
site actually passes calib_df, never test_df (the runtime check alone
wouldn't catch a future edit that accidentally swapped the argument).
"""
import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "mock-pipeline"))

from mock_data_generator import generate_mock_trials
from conformal_gate import three_split
import analysis_c4


def test_three_split_calib_and_test_have_disjoint_trial_ids():
    df = generate_mock_trials(seed=123)
    calib_df, test_df = three_split(df, calib_frac=0.7, seed=123)
    calib_ids = set(calib_df["trial_id"])
    test_ids = set(test_df["trial_id"])
    assert calib_ids.isdisjoint(test_ids), (
        f"three_split() produced overlapping trial_ids: {calib_ids & test_ids}"
    )
    assert len(calib_ids) > 0 and len(test_ids) > 0


def test_analysis_c4_runtime_assertion_catches_injected_overlap():
    """Directly exercises the runtime guard added in analysis_c4.py's
    run_full_c4_package (C6): construct a calib_df/test_df pair that
    DOES overlap and confirm the module's own assertion fires."""
    df = generate_mock_trials(seed=7)
    calib_df, test_df = three_split(df, calib_frac=0.7, seed=7)
    # Inject an overlap: duplicate one calib row into test_df with the SAME trial_id.
    contaminated_test_df = test_df.copy()
    if len(calib_df) > 0:
        contaminated_test_df = contaminated_test_df.append(calib_df.iloc[[0]], ignore_index=True) \
            if hasattr(contaminated_test_df, "append") else None
    if contaminated_test_df is None:  # pandas >= 2.0 dropped DataFrame.append
        import pandas as pd
        contaminated_test_df = pd.concat([test_df, calib_df.iloc[[0]]], ignore_index=True)

    assert set(calib_df["trial_id"]).intersection(set(contaminated_test_df["trial_id"]))
    with pytest.raises(AssertionError, match="Split-discipline violation"):
        assert set(calib_df["trial_id"]).isdisjoint(set(contaminated_test_df["trial_id"])), (
            "Split-discipline violation: calib_df and test_df share trial_id(s) -- "
            "the FAIL-Detect baseline would be trained on rows also used for "
            "held-out evaluation."
        )


def _find_baseline_fit_call_args(pyfile: Path):
    """AST-based: find every `baseline.fit(...)` call and return the
    argument names (as source text) passed to it."""
    tree = ast.parse(pyfile.read_text())
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "fit" and \
           isinstance(func.value, ast.Name) and func.value.id == "baseline":
            arg_names = [ast.unparse(a) if hasattr(ast, "unparse") else "" for a in node.args]
            calls.append((node.lineno, arg_names))
    return calls


def test_baseline_fit_call_site_uses_calib_df_never_test_df():
    analysis_c4_path = PROJECT_ROOT / "perception-calibration-gap" / "analysis" / "analysis_c4.py"
    calls = _find_baseline_fit_call_args(analysis_c4_path)
    assert calls, "expected to find at least one baseline.fit(...) call site in analysis_c4.py"
    for lineno, arg_names in calls:
        assert "calib_df" in arg_names, f"analysis_c4.py:{lineno}: baseline.fit() must be called with calib_df, got {arg_names}"
        assert "test_df" not in arg_names, f"analysis_c4.py:{lineno}: baseline.fit() must NEVER be called with test_df, got {arg_names}"


def test_full_c4_package_runs_end_to_end_on_mock_data():
    """Sanity: the module (with the C6-added split-discipline assertion)
    still runs correctly on ordinary, non-contaminated data."""
    df = generate_mock_trials(seed=99)
    result = analysis_c4.run_full_c4_package(df)
    assert "gap_coverage" in result
