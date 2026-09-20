#!/usr/bin/env python3
"""
yolo/per_object_gate.py

T5: per-object acceptance-gate evaluator. Thresholds pre-committed in
project_overnight/CRITERIA.md BEFORE this was implemented (precision >
0.5, recall > 0.5, AP50 > 0.5 per object -- deliberately modest, this is
a code-path test on synthetic data, not a real-detector quality claim).

THE CORE RULE THIS FILE EXISTS TO ENFORCE: an aggregate/mAP-only number
is NEVER sufficient -- the gate must fail if even ONE object's
precision/recall/AP50 falls short, regardless of how good the aggregate
looks. test_per_object_gate.py proves this with a constructed case where
the aggregate passes but one class doesn't.

Input format: a dict {object_id: {"precision": float, "recall": float,
"ap50": float}}, matching what ultralytics' val() results (results.box.p,
results.box.r, results.box.ap50, per class) can be reshaped into --
reshaping the real ultralytics output is NOT done here (this file only
implements the gate logic on an already-per-class dict), since no real
training/validation run exists yet per this project's stated scope
(software-only, no real detector claims).
"""
from __future__ import annotations

from dataclasses import dataclass, field

PRECISION_THRESHOLD = 0.5
RECALL_THRESHOLD = 0.5
AP50_THRESHOLD = 0.5


@dataclass
class ObjectGateResult:
    object_id: str
    precision: float
    recall: float
    ap50: float
    passed: bool
    failure_reasons: list = field(default_factory=list)


@dataclass
class GateReport:
    per_object: dict  # object_id -> ObjectGateResult
    all_passed: bool
    failed_objects: list


def evaluate_object(object_id: str, metrics: dict) -> ObjectGateResult:
    precision, recall, ap50 = metrics["precision"], metrics["recall"], metrics["ap50"]
    reasons = []
    if precision <= PRECISION_THRESHOLD:
        reasons.append(f"precision {precision:.3f} <= {PRECISION_THRESHOLD}")
    if recall <= RECALL_THRESHOLD:
        reasons.append(f"recall {recall:.3f} <= {RECALL_THRESHOLD}")
    if ap50 <= AP50_THRESHOLD:
        reasons.append(f"AP50 {ap50:.3f} <= {AP50_THRESHOLD}")
    return ObjectGateResult(
        object_id=object_id, precision=precision, recall=recall, ap50=ap50,
        passed=(len(reasons) == 0), failure_reasons=reasons,
    )


def evaluate_gate(per_object_metrics: dict) -> GateReport:
    """per_object_metrics: {object_id: {"precision":.., "recall":.., "ap50":..}}
    An empty dict is a hard failure (nothing was evaluated -- refuse to
    report a pass on no data), not a vacuous pass."""
    if not per_object_metrics:
        result = GateReport(per_object={}, all_passed=False, failed_objects=["<no objects evaluated>"])
        return result

    per_object = {oid: evaluate_object(oid, m) for oid, m in per_object_metrics.items()}
    failed = [oid for oid, r in per_object.items() if not r.passed]
    return GateReport(per_object=per_object, all_passed=(len(failed) == 0), failed_objects=failed)


def aggregate_map50(per_object_metrics: dict) -> float:
    """The aggregate metric that must NEVER be used alone to decide
    pass/fail -- exposed here only so tests can construct the exact
    'aggregate passes, one class fails' scenario explicitly."""
    if not per_object_metrics:
        return 0.0
    return sum(m["ap50"] for m in per_object_metrics.values()) / len(per_object_metrics)


def print_report(report: GateReport) -> None:
    print(f"{'object_id':<12}{'precision':<12}{'recall':<12}{'AP50':<12}{'PASS/FAIL':<10}")
    for oid, r in report.per_object.items():
        status = "PASS" if r.passed else "FAIL"
        print(f"{oid:<12}{r.precision:<12.3f}{r.recall:<12.3f}{r.ap50:<12.3f}{status:<10}")
        for reason in r.failure_reasons:
            print(f"    -> {reason}")
    print(f"\nOverall gate: {'PASS' if report.all_passed else 'FAIL'} "
          f"(failed objects: {report.failed_objects if report.failed_objects else 'none'})")


if __name__ == "__main__":
    demo_metrics = {
        "apple": {"precision": 0.7, "recall": 0.65, "ap50": 0.72},
        "purse": {"precision": 0.55, "recall": 0.6, "ap50": 0.58},
        "glass": {"precision": 0.3, "recall": 0.9, "ap50": 0.95},  # low precision -> should FAIL despite high AP50
    }
    print(f"Aggregate mAP50 (NEVER sufficient alone): {aggregate_map50(demo_metrics):.3f}")
    report = evaluate_gate(demo_metrics)
    print_report(report)
