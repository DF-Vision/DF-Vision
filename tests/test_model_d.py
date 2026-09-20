"""
tests/test_model_d.py

Hard rule 5 tests: stub/unreal prg rows are excluded with a loud logged
count; Model D refuses to run when zero real rows exist; the Gap-only
path runs with prg columns entirely absent.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "analysis"))

import numpy as np
import pandas as pd
import pytest

from model_d import (
    extract_trial_prg_signal, filter_real_prg_rows, run_model_d, run_gap_only,
    ModelDRefused, _is_enabled,
)

RANDOM_STATE = 20260920


def make_synthetic_trials(n=60, real_fraction=0.5, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    objects = ["apple", "purse", "glass"]
    trials = []
    for i in range(n):
        obj = objects[i % 3]
        true_occlusion = float(rng.uniform(0, 0.8))
        gap = float(np.clip(0.3 * true_occlusion + rng.normal(0, 0.1), 0, 1))
        logit = -1.0 + 2.0 * true_occlusion + 1.5 * gap
        failure = int(rng.uniform() < 1 / (1 + np.exp(-logit)))

        is_real = rng.uniform() < real_fraction
        if is_real:
            trial = {
                "trial_id": i, "object_id": obj, "true_occlusion": true_occlusion,
                "Gap_initial": gap, "failure": failure,
                "prg_is_stub": False,
                "prg_edges": [{"neighbor_name": "x", "prg_edge_error": float(rng.uniform(0, 1))}],
            }
        else:
            # Mix of exclusion reasons, deliberately, to test all 3 counters.
            reason = i % 3
            if reason == 0:
                trial = {"trial_id": i, "object_id": obj, "true_occlusion": true_occlusion,
                          "Gap_initial": gap, "failure": failure,
                          "prg_is_stub": True, "prg_edges": [{"prg_edge_error": 0.5}]}
            elif reason == 1:
                trial = {"trial_id": i, "object_id": obj, "true_occlusion": true_occlusion,
                          "Gap_initial": gap, "failure": failure,
                          "prg_is_stub": False, "prg_edges": []}
            else:
                trial = {"trial_id": i, "object_id": obj, "true_occlusion": true_occlusion,
                          "Gap_initial": gap, "failure": failure,
                          "prg_is_stub": False, "prg_edges": [{"prg_edge_error": float("nan")}]}
        trials.append(trial)
    return trials


def test_extract_trial_prg_signal_real_row():
    trial = {"prg_is_stub": False, "prg_edges": [{"prg_edge_error": 0.3}, {"prg_edge_error": 0.7}]}
    assert extract_trial_prg_signal(trial) == pytest.approx(0.7)  # max aggregation


def test_extract_trial_prg_signal_excludes_stub():
    trial = {"prg_is_stub": True, "prg_edges": [{"prg_edge_error": 0.3}]}
    assert extract_trial_prg_signal(trial) is None


def test_extract_trial_prg_signal_excludes_missing_edges():
    trial = {"prg_is_stub": False, "prg_edges": []}
    assert extract_trial_prg_signal(trial) is None
    trial2 = {"prg_is_stub": False, "prg_edges": None}
    assert extract_trial_prg_signal(trial2) is None


def test_extract_trial_prg_signal_excludes_all_nan():
    trial = {"prg_is_stub": False, "prg_edges": [{"prg_edge_error": float("nan")}]}
    assert extract_trial_prg_signal(trial) is None


def test_extract_trial_prg_signal_excludes_missing_stub_flag():
    """prg_is_stub missing entirely (None) must NOT be treated as real --
    only an explicit False counts."""
    trial = {"prg_edges": [{"prg_edge_error": 0.5}]}
    assert extract_trial_prg_signal(trial) is None


def test_filter_real_prg_rows_counts_are_loud_and_correct(capsys):
    trials = make_synthetic_trials(n=60, real_fraction=0.5)
    df = filter_real_prg_rows(trials)
    captured = capsys.readouterr()
    assert "EXCLUDED" in captured.out
    assert "stub=" in captured.out
    n_expected_real = sum(1 for t in trials if extract_trial_prg_signal(t) is not None)
    assert len(df) == n_expected_real
    assert len(df) < len(trials)  # confirms some rows really were excluded


def test_model_d_refuses_when_disabled():
    assert _is_enabled() is False  # committed default
    trials = make_synthetic_trials(n=30, real_fraction=1.0)
    with pytest.raises(ModelDRefused, match="disabled"):
        run_model_d(trials, force_enabled=False)


def test_model_d_refuses_on_zero_real_rows():
    trials = make_synthetic_trials(n=20, real_fraction=0.0)  # all excluded
    with pytest.raises(ModelDRefused, match="[Zz]ero real"):
        run_model_d(trials, force_enabled=True)


def test_model_d_runs_with_force_enabled_and_real_rows():
    trials = make_synthetic_trials(n=80, real_fraction=0.7)
    result = run_model_d(trials, force_enabled=True)
    assert "model_d" in result
    assert result["n_real_rows"] > 0
    assert result["n_real_rows"] < len(trials)  # confirms exclusion actually happened


def test_gap_only_runs_without_prg_columns():
    """The Gap-only path must work when prg columns are ENTIRELY ABSENT
    from the dataframe, not just None-valued."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = 60
    df = pd.DataFrame({
        "object_id": ["apple", "purse", "glass"] * (n // 3),
        "true_occlusion": rng.uniform(0, 0.8, n),
        "Gap_initial": rng.uniform(0, 0.5, n),
    })
    df["failure"] = (rng.uniform(size=n) < 0.3).astype(int)
    assert "prg_edge_error" not in df.columns
    assert "prg_is_stub" not in df.columns
    result = run_gap_only(df)  # must not raise, must not need prg columns
    assert "model_b_gap_only" in result


def test_model_d_config_file_default_is_disabled():
    import json
    config_path = PROJECT_ROOT / "perception-calibration-gap" / "analysis" / "model_d_config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        assert cfg.get("enabled", False) is False, (
            "model_d_config.json must default to disabled (rule 8/10)"
        )
