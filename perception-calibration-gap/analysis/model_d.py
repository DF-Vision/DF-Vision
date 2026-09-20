#!/usr/bin/env python3
"""
analysis/model_d.py

Model D (`failure ~ true_occlusion + Gap_initial + prg_edge_error + C(object_id)`)
and the stub-exclusion rule hard rule 5 requires. GUARDED, OFF BY DEFAULT:

    MODEL_D_ENABLED = False   (module-level constant, also overridable via
                                analysis/model_d_config.json's "enabled" key)

Nothing in this file runs automatically when imported -- `run_model_d()`
itself refuses unless explicitly enabled, and separately refuses if zero
REAL (non-stub) prg rows exist after filtering.

STUB-EXCLUSION RULE (hard rule 5): a trial's PRG signal is REAL only if
`prg_is_stub is False` AND `prg_edges` is a non-empty list AND at least
one edge in it has a `prg_edge_error` that is not None/NaN. Every other
trial (prg_is_stub True, prg_edges None/[], or all-NaN edges) is EXCLUDED
from Model D with a loud logged count -- never silently dropped, never
silently imputed.

Per-trial prg_edge_error aggregation: a trial can have >1 adjacent-object
edge (see SCHEMA.md/logger.py's prg_edges list design). Model D needs one
scalar per trial. This module uses MAX prg_edge_error across a trial's
real edges (the most conservative "worst boundary reliability" signal,
consistent with the two-signal conformal gate's own max-combination
design in mock-pipeline/run_trial.py's OnlineGateAdapter/PREREG.md's
C4-ext amendment) -- documented here explicitly since it's a modeling
choice, not the only defensible one.

The Gap-only path (analysis_c1.py's existing Models A/B/C) is
UNAFFECTED by anything in this file and runs with prg columns absent
entirely -- proven by test_model_d.py's test_gap_only_runs_without_prg_columns.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)

MODEL_D_ENABLED = False  # hard default OFF -- rule 8/10
_CONFIG_PATH = Path(__file__).parent / "model_d_config.json"


def _is_enabled() -> bool:
    if not _CONFIG_PATH.exists():
        return MODEL_D_ENABLED
    try:
        cfg = json.loads(_CONFIG_PATH.read_text())
        return bool(cfg.get("enabled", MODEL_D_ENABLED))
    except (json.JSONDecodeError, OSError):
        return MODEL_D_ENABLED


class ModelDRefused(Exception):
    """Raised whenever run_model_d() refuses to fit -- disabled, or zero
    real rows. Callers must treat this as a hard stop."""


def extract_trial_prg_signal(trial: dict) -> Optional[float]:
    """Returns the trial's aggregated (max) real prg_edge_error, or None
    if this trial has no usable real PRG signal (stub, missing, or
    all-NaN edges) -- the per-trial half of the stub-exclusion rule."""
    if trial.get("prg_is_stub", None) is not False:
        return None  # None/True both excluded -- only an EXPLICIT False counts as real
    edges = trial.get("prg_edges") or []
    if not edges:
        return None
    real_values = [
        e["prg_edge_error"] for e in edges
        if e.get("prg_edge_error") is not None and not (
            isinstance(e["prg_edge_error"], float) and np.isnan(e["prg_edge_error"])
        )
    ]
    if not real_values:
        return None
    return float(max(real_values))


def filter_real_prg_rows(trials: list[dict]) -> pd.DataFrame:
    """Builds a DataFrame with one row per trial that has a REAL PRG
    signal, LOUDLY logging (print + logger.warning, not just a silent
    debug line) how many trials were excluded and why."""
    rows = []
    n_stub, n_missing_edges, n_all_nan = 0, 0, 0
    for trial in trials:
        signal = extract_trial_prg_signal(trial)
        if signal is None:
            if trial.get("prg_is_stub", None) is not False:
                n_stub += 1
            elif not (trial.get("prg_edges") or []):
                n_missing_edges += 1
            else:
                n_all_nan += 1
            continue
        row = dict(trial)
        row["prg_edge_error_agg"] = signal
        rows.append(row)

    n_total = len(trials)
    n_real = len(rows)
    n_excluded = n_total - n_real
    msg = (
        f"[model_d.filter_real_prg_rows] {n_total} trials in -> {n_real} REAL rows, "
        f"{n_excluded} EXCLUDED (stub={n_stub}, missing/empty prg_edges={n_missing_edges}, "
        f"all-NaN edges={n_all_nan})."
    )
    print(msg)
    logger.warning(msg)

    return pd.DataFrame(rows)


def run_gap_only(df: pd.DataFrame) -> dict:
    """The Gap-only path (C1, no prg dependency at all). Runs correctly
    whether or not the input df has any prg_* columns -- proven by
    test_model_d.py's test_gap_only_runs_without_prg_columns, which
    passes a df with prg columns dropped entirely."""
    model_b = smf.logit("failure ~ true_occlusion + Gap_initial + C(object_id)", data=df).fit(disp=0)
    return {"model_b_gap_only": model_b}


def run_model_d(trials: list[dict], force_enabled: bool = False) -> dict:
    """
    trials: raw logged trial dicts (e.g. from TrialLogger.load_trials()).
    force_enabled: in-memory override for testing ONLY -- never set True
    by anything other than this module's own test suite; does not touch
    analysis/model_d_config.json on disk.

    Refuses (ModelDRefused) if:
      - not enabled (MODEL_D_ENABLED False and no config override), unless force_enabled
      - zero real (non-stub) PRG rows after filtering
    """
    if not (force_enabled or _is_enabled()):
        raise ModelDRefused(
            "Model D is disabled (MODEL_D_ENABLED=False, no config override). "
            "This is the default per hard rule 8/10 -- Model D / the two-signal gate "
            "must never run silently. Set analysis/model_d_config.json's "
            '{"enabled": true} explicitly to enable, or pass force_enabled=True for tests.'
        )

    real_df = filter_real_prg_rows(trials)
    if len(real_df) == 0:
        raise ModelDRefused(
            "Zero real (non-stub) PRG rows after filtering -- refusing to fit Model D. "
            "(hard rule 5: Model D must refuse to run when zero real rows exist, not "
            "silently fit on stub data or an empty frame.)"
        )

    model_d = smf.logit(
        "failure ~ true_occlusion + Gap_initial + prg_edge_error_agg + C(object_id)",
        data=real_df,
    ).fit(disp=0)
    return {"model_d": model_d, "n_real_rows": len(real_df), "real_df": real_df}


if __name__ == "__main__":
    print(f"Model D enabled: {_is_enabled()}")
    print("This module does nothing further when run directly -- see "
          "test_model_d.py or call run_model_d(trials) from analysis code.")
