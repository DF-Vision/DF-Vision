"""
summarize_mujoco_trials.py

Reads trials_mujoco_smoke.jsonl (produced by mujoco_day3.py, Phase 0.25)
and prints a clean, presentable summary -- the "small simulated dataset"
deliverable for the mentor's MuJoCo-before-hardware requirement.

IMPORTANT SCOPE NOTE (carried from RUNNING_NOTES.md / mujoco_day3.py's own
docstrings): this dataset is INFRASTRUCTURE VALIDATION ONLY. It uses a
synthetic/placeholder detector bbox (the target's own projected AABB, not
a real YOLO/open-vocab detector call) and a scripted/mock policy stand-in,
not the real frozen manipulation policy. It is never to be cited as
evidence for C1/C4/C1-ext/C4-ext statistical results, and never merged
into the real trial dataset. Its sole purpose is to demonstrate the
pipeline (detector-interface -> TSDF -> Gap -> policy-call -> logging)
runs end-to-end and produces sane, non-degenerate values before real
hardware integration begins.

Usage:
    python3 summarize_mujoco_trials.py [path/to/trials_mujoco_smoke.jsonl]

Default path assumes it's run from inside the container at /workspace,
matching mujoco_day3.py's own LOG_PATH.
"""

import json
import sys
from pathlib import Path


DEFAULT_LOG_PATH = "/workspace/mock-pipeline/data/trials_mujoco_smoke.jsonl"

# Fields we expect in every trial record, for a quick completeness check
# separate from mujoco_day3.py's own bad_fields check (this is a display-
# layer sanity check, not a re-validation of the schema itself).
DISPLAY_FIELDS = [
    "trial_id", "true_occlusion", "measured_true_visibility",
    "perceived_visibility_initial", "Gap_initial", "outcome",
]


def load_trials(path: str) -> list[dict]:
    trials = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                trials.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: could not parse line {line_num}: {e}")
    return trials


def summarize(trials: list[dict]) -> None:
    if not trials:
        print("No trials loaded -- nothing to summarize.")
        return

    n = len(trials)
    gaps = [t.get("Gap_initial") for t in trials if t.get("Gap_initial") is not None]
    true_vis = [t.get("measured_true_visibility") for t in trials
                if t.get("measured_true_visibility") is not None]
    perceived_vis = [t.get("perceived_visibility_initial") for t in trials
                      if t.get("perceived_visibility_initial") is not None]
    outcomes = [t.get("outcome") for t in trials]

    outcome_counts: dict[str, int] = {}
    for o in outcomes:
        outcome_counts[o] = outcome_counts.get(o, 0) + 1

    print("=" * 70)
    print("MuJoCo Simulated Trial Dataset -- Summary")
    print("=" * 70)
    print(f"Total trials: {n}")
    print()
    print("--- Ranges (sanity check: nothing degenerate, real variation) ---")
    if true_vis:
        print(f"  true_visibility:      min={min(true_vis):.4f}  "
              f"max={max(true_vis):.4f}  range={max(true_vis)-min(true_vis):.4f}")
    if perceived_vis:
        print(f"  perceived_visibility: min={min(perceived_vis):.4f}  "
              f"max={max(perceived_vis):.4f}  range={max(perceived_vis)-min(perceived_vis):.4f}")
    if gaps:
        print(f"  Gap_initial:          min={min(gaps):.4f}  "
              f"max={max(gaps):.4f}  range={max(gaps)-min(gaps):.4f}")
    print()
    print("--- Outcome distribution ---")
    for outcome, count in sorted(outcome_counts.items()):
        print(f"  {outcome:12s}: {count}/{n}")
    print()
    print("--- Per-trial table ---")
    header = f"{'trial':>5} {'true_occ':>9} {'true_vis':>9} {'perc_vis':>9} {'Gap':>7} {'outcome':>9}"
    print(header)
    print("-" * len(header))
    for t in trials:
        print(
            f"{t.get('trial_id', '?'):>5} "
            f"{t.get('true_occlusion', float('nan')):>9.4f} "
            f"{t.get('measured_true_visibility', float('nan')):>9.4f} "
            f"{t.get('perceived_visibility_initial', float('nan')):>9.4f} "
            f"{t.get('Gap_initial', float('nan')):>7.4f} "
            f"{str(t.get('outcome', '?')):>9}"
        )
    print()

    # Completeness check -- flag any trial missing a display field, separate
    # from mujoco_day3.py's own schema validation.
    incomplete = []
    for t in trials:
        missing = [f for f in DISPLAY_FIELDS if t.get(f) is None]
        if missing:
            incomplete.append((t.get("trial_id", "?"), missing))
    if incomplete:
        print("--- WARNING: trials with missing display fields ---")
        for trial_id, missing in incomplete:
            print(f"  trial {trial_id}: missing {missing}")
    else:
        print(f"All {n} trials have complete display fields. OK.")

    print()
    print("=" * 70)
    print("SCOPE REMINDER: this is infrastructure validation only (synthetic")
    print("detector bbox, mock/scripted policy stand-in). NOT real trial")
    print("data. Never cited as evidence for C1/C4/C1-ext/C4-ext results.")
    print("=" * 70)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG_PATH
    if not Path(path).exists():
        print(f"ERROR: {path} does not exist.")
        print("Pass the correct path as an argument, e.g.:")
        print("  python3 summarize_mujoco_trials.py /path/to/trials_mujoco_smoke.jsonl")
        sys.exit(1)

    trials = load_trials(path)
    summarize(trials)


if __name__ == "__main__":
    main()
