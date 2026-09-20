#!/usr/bin/env python3
"""
analysis/data_quality_report.py

T9: trials per object x condition x repeat, success/failure counts,
stub-vs-real prg rows, exclusions, session_id vs object_id confounding
check, and a controller-freeze check (fails if controller_git_hash
changes mid-dataset without an explicit new-dataset tag file present).

Reads a TrialLogger-format JSONL log; does not modify it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mock-pipeline"))
from logger import TrialLogger  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_d import extract_trial_prg_signal  # noqa: E402


def load_trials_df(log_path: str) -> pd.DataFrame:
    logger_obj = TrialLogger(log_path)
    trials = logger_obj.load_trials(strict=False)
    return pd.DataFrame(trials), trials


def counts_by_object_condition_repeat(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["object_id", "scene_condition", "repeat_id"] if c in df.columns]
    if not cols or df.empty:
        return pd.DataFrame()
    return df.groupby(cols, dropna=False).size().reset_index(name="n_trials")


def success_failure_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "object_id" not in df.columns or "failure" not in df.columns or df.empty:
        return pd.DataFrame()
    return df.groupby("object_id")["failure"].agg(
        n_trials="count", n_failures="sum", n_successes=lambda s: (s == 0).sum()
    ).reset_index()


def prg_stub_vs_real_counts(trials: list[dict]) -> dict:
    n_real = sum(1 for t in trials if extract_trial_prg_signal(t) is not None)
    n_stub = sum(1 for t in trials if t.get("prg_is_stub") is True)
    n_no_prg = sum(1 for t in trials if not t.get("prg_edges"))
    return {"n_trials": len(trials), "n_real_prg_rows": n_real,
            "n_stub_prg_rows": n_stub, "n_trials_without_prg_edges": n_no_prg}


def session_object_confounding_check(df: pd.DataFrame) -> dict:
    """Flags if any session_id is associated with only ONE object_id --
    a design confound that would make session effects indistinguishable
    from object effects (exactly what the rotated-object-order scheduler
    below exists to prevent)."""
    if "session_id" not in df.columns or "object_id" not in df.columns or df.empty:
        return {"checked": False, "reason": "missing session_id/object_id columns or empty df"}
    per_session_objects = df.groupby("session_id")["object_id"].nunique()
    confounded_sessions = per_session_objects[per_session_objects <= 1].index.tolist()
    return {
        "checked": True,
        "n_sessions": len(per_session_objects),
        "confounded_sessions": confounded_sessions,
        "confounding_detected": len(confounded_sessions) > 0,
    }


NEW_DATASET_TAG_SUFFIX = ".NEW_DATASET_TAG"


def controller_freeze_check(df: pd.DataFrame, log_path: str) -> dict:
    """Fails (returns ok=False) if controller_git_hash changes mid-dataset
    UNLESS a sibling '<log_path>.NEW_DATASET_TAG' file exists whose
    content names the new hash -- an explicit, human-authored marker
    that the change was intentional (a new dataset epoch), not silent
    controller drift mid-collection."""
    if "controller_git_hash" not in df.columns or df.empty:
        return {"checked": False, "reason": "no controller_git_hash column or empty df"}
    hashes = df["controller_git_hash"].dropna().unique().tolist()
    if len(hashes) <= 1:
        return {"checked": True, "ok": True, "hashes_seen": hashes}

    tag_path = Path(str(log_path) + NEW_DATASET_TAG_SUFFIX)
    if tag_path.exists():
        return {
            "checked": True, "ok": True, "hashes_seen": hashes,
            "note": f"Multiple controller hashes found but {tag_path} exists -- "
                    f"treated as an intentional new-dataset tag, not a freeze violation.",
        }
    return {
        "checked": True, "ok": False, "hashes_seen": hashes,
        "error": f"controller_git_hash changed mid-dataset ({hashes}) with no "
                 f"{tag_path.name} tag present. This is a hard failure per rule 9 "
                 f"(train/serve-style consistency, applied to controller version): "
                 f"either revert to one controller version for this dataset, or "
                 f"create {tag_path} explicitly acknowledging a new dataset epoch.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=str)
    args = parser.parse_args()

    df, trials = load_trials_df(args.log)
    print(f"Loaded {len(trials)} trials from {args.log}")

    print("\n--- Trials per object x condition x repeat ---")
    print(counts_by_object_condition_repeat(df).to_string(index=False) if len(df) else "(no data)")

    print("\n--- Success/failure counts per object ---")
    print(success_failure_counts(df).to_string(index=False) if len(df) else "(no data)")

    print("\n--- PRG stub-vs-real ---")
    print(json.dumps(prg_stub_vs_real_counts(trials), indent=2))

    print("\n--- Session/object confounding check ---")
    confound = session_object_confounding_check(df)
    print(json.dumps(confound, indent=2))
    if confound.get("confounding_detected"):
        print("[WARNING] Some sessions are confounded with a single object_id -- "
              "session effects and object effects will not be separable for those rows.")

    print("\n--- Controller-freeze check ---")
    freeze = controller_freeze_check(df, args.log)
    print(json.dumps(freeze, indent=2))
    if freeze.get("checked") and not freeze.get("ok", True):
        print("[FAIL] Controller-freeze check failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
