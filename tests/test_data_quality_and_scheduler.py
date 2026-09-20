"""
tests/test_data_quality_and_scheduler.py

T9: data_quality_report.py (confounding check, controller-freeze check,
stub-vs-real counts) and session_scheduler.py (rotated object order).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "analysis"))

import pandas as pd
import pytest

from data_quality_report import (
    session_object_confounding_check, controller_freeze_check, prg_stub_vs_real_counts,
    counts_by_object_condition_repeat,
)
from session_scheduler import rotated_order, build_schedule, verify_no_starting_object_repeats_within_cycle


# --- data_quality_report.py ---------------------------------------------------

def test_confounding_detected_when_session_has_one_object():
    df = pd.DataFrame({
        "session_id": [0, 0, 0, 1, 1, 1],
        "object_id": ["apple", "apple", "apple", "purse", "purse", "purse"],
    })
    result = session_object_confounding_check(df)
    assert result["confounding_detected"] is True
    assert set(result["confounded_sessions"]) == {0, 1}


def test_no_confounding_when_sessions_mix_objects():
    df = pd.DataFrame({
        "session_id": [0, 0, 0, 1, 1, 1],
        "object_id": ["apple", "purse", "glass", "purse", "glass", "apple"],
    })
    result = session_object_confounding_check(df)
    assert result["confounding_detected"] is False


def test_controller_freeze_ok_with_one_hash():
    df = pd.DataFrame({"controller_git_hash": ["abc123"] * 10})
    result = controller_freeze_check(df, "/tmp/does_not_matter.jsonl")
    assert result["ok"] is True


def test_controller_freeze_fails_on_mid_dataset_change_without_tag(tmp_path):
    df = pd.DataFrame({"controller_git_hash": ["abc123"] * 5 + ["def456"] * 5})
    log_path = str(tmp_path / "trials.jsonl")
    result = controller_freeze_check(df, log_path)
    assert result["ok"] is False
    assert "abc123" in str(result["hashes_seen"])
    assert "def456" in str(result["hashes_seen"])


def test_controller_freeze_passes_with_explicit_new_dataset_tag(tmp_path):
    df = pd.DataFrame({"controller_git_hash": ["abc123"] * 5 + ["def456"] * 5})
    log_path = tmp_path / "trials.jsonl"
    tag_path = tmp_path / "trials.jsonl.NEW_DATASET_TAG"
    tag_path.write_text("def456 -- intentional controller update on 2026-09-22\n")
    result = controller_freeze_check(df, str(log_path))
    assert result["ok"] is True


def test_prg_stub_vs_real_counts():
    trials = [
        {"prg_is_stub": False, "prg_edges": [{"prg_edge_error": 0.3}]},
        {"prg_is_stub": True, "prg_edges": [{"prg_edge_error": 0.5}]},
        {"prg_is_stub": False, "prg_edges": []},
    ]
    counts = prg_stub_vs_real_counts(trials)
    assert counts["n_trials"] == 3
    assert counts["n_real_prg_rows"] == 1
    assert counts["n_stub_prg_rows"] == 1
    assert counts["n_trials_without_prg_edges"] == 1


def test_counts_by_object_condition_repeat():
    df = pd.DataFrame({
        "object_id": ["apple", "apple", "purse"],
        "scene_condition": ["contact", "contact", "separate"],
        "repeat_id": [0, 1, 0],
    })
    out = counts_by_object_condition_repeat(df)
    assert len(out) == 3  # apple/contact/0, apple/contact/1, purse/separate/0
    assert out["n_trials"].sum() == 3


# --- session_scheduler.py -----------------------------------------------------

def test_rotated_order_cycles_correctly():
    objects = ["apple", "purse", "glass"]
    assert rotated_order(objects, 0) == ["apple", "purse", "glass"]
    assert rotated_order(objects, 1) == ["purse", "glass", "apple"]
    assert rotated_order(objects, 2) == ["glass", "apple", "purse"]
    assert rotated_order(objects, 3) == ["apple", "purse", "glass"]  # cycles back


def test_every_object_starts_exactly_once_per_cycle():
    objects = ["apple", "purse", "glass"]
    schedule = build_schedule(objects, n_days=9, trials_per_object_per_day=3)
    assert verify_no_starting_object_repeats_within_cycle(schedule, objects) is True
    starts = [d["object_order"][0] for d in schedule[:3]]
    assert set(starts) == set(objects)


def test_schedule_total_trials_correct():
    objects = ["apple", "purse", "glass"]
    schedule = build_schedule(objects, n_days=2, trials_per_object_per_day=5)
    for day in schedule:
        assert day["total_trials"] == 15
        assert len(day["object_order"]) == 3
