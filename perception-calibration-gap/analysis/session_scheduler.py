#!/usr/bin/env python3
"""
analysis/session_scheduler.py

T9: session-blocking scheduler. Outputs a rotated object order per day,
so no single session/day is confounded with a single object_id (the
exact confound data_quality_report.py's session_object_confounding_check
flags). Uses a Latin-square-style rotation: with k objects, day i uses
the object order [obj[(i+j) % k] for j in range(k)], guaranteeing every
object appears in every rotation-position across k consecutive days and
no day repeats the same starting object as another day within one full
cycle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rotated_order(objects: list[str], day_index: int) -> list[str]:
    k = len(objects)
    if k == 0:
        return []
    return [objects[(day_index + j) % k] for j in range(k)]


def build_schedule(objects: list[str], n_days: int, trials_per_object_per_day: int) -> list[dict]:
    schedule = []
    for day in range(n_days):
        order = rotated_order(objects, day)
        schedule.append({
            "day": day,
            "session_id": day,
            "object_order": order,
            "trials_per_object": trials_per_object_per_day,
            "total_trials": trials_per_object_per_day * len(order),
        })
    return schedule


def verify_no_starting_object_repeats_within_cycle(schedule: list[dict], objects: list[str]) -> bool:
    """Sanity check: within one full k-day cycle, every object starts
    exactly one day (proves the rotation is a genuine Latin square, not
    a bug that accidentally repeats)."""
    k = len(objects)
    if k == 0 or len(schedule) < k:
        return True
    starts = [day["object_order"][0] for day in schedule[:k]]
    return len(set(starts)) == k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=str, default="apple,purse,glass")
    parser.add_argument("--n-days", type=int, default=9)
    parser.add_argument("--trials-per-object-per-day", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    objects = args.objects.split(",")
    schedule = build_schedule(objects, args.n_days, args.trials_per_object_per_day)

    ok = verify_no_starting_object_repeats_within_cycle(schedule, objects)
    print(f"Latin-square rotation check: {'OK' if ok else 'FAILED -- this is a bug, report it'}")
    for day in schedule:
        print(f"Day {day['day']}: {day['object_order']} "
              f"({day['trials_per_object']} trials/object, {day['total_trials']} total)")

    if args.output:
        Path(args.output).write_text(json.dumps(schedule, indent=2))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
