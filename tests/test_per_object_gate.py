"""
tests/test_per_object_gate.py

T5: proves the per-object acceptance gate rejects a result where the
AGGREGATE passes but one object's metrics don't -- the core requirement
("an aggregate metric alone never passes").
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "yolo"))

from per_object_gate import evaluate_gate, evaluate_object, aggregate_map50


def test_all_objects_passing_gives_overall_pass():
    metrics = {
        "apple": {"precision": 0.7, "recall": 0.7, "ap50": 0.75},
        "purse": {"precision": 0.6, "recall": 0.65, "ap50": 0.6},
        "glass": {"precision": 0.55, "recall": 0.55, "ap50": 0.58},
    }
    report = evaluate_gate(metrics)
    assert report.all_passed is True
    assert report.failed_objects == []


def test_aggregate_passes_but_one_object_fails_gate_still_fails():
    """THE core test: construct a case where mAP50 (aggregate) is well
    above threshold, but one object's own metrics are not."""
    metrics = {
        "apple": {"precision": 0.9, "recall": 0.9, "ap50": 0.95},
        "purse": {"precision": 0.9, "recall": 0.9, "ap50": 0.95},
        "glass": {"precision": 0.2, "recall": 0.2, "ap50": 0.3},  # clearly failing
    }
    agg = aggregate_map50(metrics)
    assert agg > 0.5, f"test setup: aggregate mAP50 ({agg:.3f}) must be well above threshold"

    report = evaluate_gate(metrics)
    assert report.all_passed is False, (
        "Gate must FAIL when even one object fails, even though the aggregate "
        f"mAP50 ({agg:.3f}) is comfortably above 0.5 -- an aggregate-only check "
        "would have wrongly passed this."
    )
    assert "glass" in report.failed_objects
    assert "apple" not in report.failed_objects
    assert "purse" not in report.failed_objects


def test_high_precision_recall_but_low_ap50_still_fails():
    metrics = {"apple": {"precision": 0.9, "recall": 0.9, "ap50": 0.3}}
    report = evaluate_gate(metrics)
    assert report.all_passed is False
    assert any("AP50" in r for r in report.per_object["apple"].failure_reasons)


def test_empty_metrics_is_a_hard_failure_not_vacuous_pass():
    report = evaluate_gate({})
    assert report.all_passed is False


def test_evaluate_object_reports_all_failing_dimensions():
    result = evaluate_object("apple", {"precision": 0.1, "recall": 0.1, "ap50": 0.1})
    assert result.passed is False
    assert len(result.failure_reasons) == 3  # precision, recall, AND ap50 all flagged


def test_threshold_boundary_is_strict_greater_than():
    """Exactly at the threshold does not pass (thresholds are '>', not '>=')."""
    result = evaluate_object("apple", {"precision": 0.5, "recall": 0.9, "ap50": 0.9})
    assert result.passed is False
    assert any("precision" in r for r in result.failure_reasons)
