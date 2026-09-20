#!/usr/bin/env python3
"""
c7_run_mutations.py

C7: defines and runs the 30+ mutations across the 10 named targets. See
project_closure/CRITERIA.md's C7 section for the target list and method.
Run: python3 project_closure/scripts/c7_run_mutations.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mutation_test_harness import Mutation, run_all, print_table

CAL = "perception-calibration-gap/calibration/compute_extrinsic_calibration.py"
CAL_TEST = "tests/test_extrinsic_calibration.py"

GRASP = "perception-calibration-gap/grasp/lookup_table_executor.py"
GRASP_TEST = "tests/test_grasp_lookup_table.py"

MARKER = "perception-calibration-gap/calibration/capture_marker_points.py"
MARKER_TEST = "tests/test_marker_size_config.py"

HANDCALC = "perception-calibration-gap/hand_calc_validation.py"
HANDCALC_TEST = "tests/test_hand_calc_validation.py"

MODELD = "perception-calibration-gap/analysis/model_d.py"
MODELD_TEST = "tests/test_model_d.py"

SCHED = "perception-calibration-gap/analysis/session_scheduler.py"
SCHED_TEST = "tests/test_data_quality_and_scheduler.py"

DQR = "perception-calibration-gap/analysis/data_quality_report.py"
DQR_TEST = "tests/test_data_quality_and_scheduler.py"

DEPTH = "perception-calibration-gap/real_camera/depth_io.py"
DEPTH_TEST = "tests/test_replay_mode.py"


MUTATIONS = [
    # --- 1. Calibration solver + refusal rules ---
    Mutation("1-calibration", CAL, CAL_TEST,
             "flip units guard comparison (> to <)",
             "if max_mag > UNIT_GUARD_MAX_M:", "if max_mag < UNIT_GUARD_MAX_M:"),
    Mutation("1-calibration", CAL, CAL_TEST,
             "drop degeneracy refusal check (always False)",
             "if ratio < DEGENERACY_RATIO_THRESHOLD:", "if ratio < DEGENERACY_RATIO_THRESHOLD and False:"),
    Mutation("1-calibration", CAL, CAL_TEST,
             "flip Kabsch reflection-correction sign",
             "d = np.sign(np.linalg.det(Vt.T @ U.T))", "d = -np.sign(np.linalg.det(Vt.T @ U.T))"),

    # --- 2. Lookup-table / out-of-reach logic ---
    Mutation("2-lookup-table", GRASP, GRASP_TEST,
             "loosen out-of-reach threshold by 100x (extrapolates)",
             "if distance > table.max_bin_distance_m:", "if distance > table.max_bin_distance_m * 100:"),
    Mutation("2-lookup-table", GRASP, GRASP_TEST,
             "nearest_bin picks FARTHEST bin instead of nearest",
             "idx = int(np.argmin(distances))", "idx = int(np.argmax(distances))"),
    Mutation("2-lookup-table", GRASP, GRASP_TEST,
             "joint-limit check off-by-one (excludes upper bound)",
             "if not (lo <= value <= hi):", "if not (lo <= value < hi):"),

    # --- 3. Dry-run / --i-am-supervising guard ---
    Mutation("3-dry-run-guard", GRASP, GRASP_TEST,
             "drop the --i-am-supervising refusal entirely",
             "if not dry_run and not i_am_supervising:", "if False:"),
    Mutation("3-dry-run-guard", GRASP, GRASP_TEST,
             "invert dry_run branch (would call motor in dry-run mode!)",
             "    if dry_run:\n        print(f\"[DRY RUN]", "    if not dry_run:\n        print(f\"[DRY RUN]"),
    Mutation("3-dry-run-guard", GRASP, GRASP_TEST,
             "flip supervising condition (AND -> OR polarity swap)",
             "if not dry_run and not i_am_supervising:", "if not dry_run and i_am_supervising:"),

    # --- 4. Marker-size config read ---
    Mutation("4-marker-size", MARKER, MARKER_TEST,
             "wrong unit conversion divisor (cm->m uses /10 instead of /100)",
             "out[int(mid)] = float(size_cm) / 100.0", "out[int(mid)] = float(size_cm) / 10.0"),
    Mutation("4-marker-size", MARKER, MARKER_TEST,
             "marker_id kept as string instead of int",
             "out[int(mid)] = float(size_cm) / 100.0", "out[mid] = float(size_cm) / 100.0"),
    Mutation("4-marker-size", MARKER, MARKER_TEST,
             "split on wrong separator (drops multi-pair parsing)",
             'for pair in spec.split(","):', "for pair in spec.split(';'):"),

    # --- 5. Template-CSV guard (hand_calc_validation.py) ---
    Mutation("5-template-guard", HANDCALC, HANDCALC_TEST,
             "remove 'REPLACE with' detection substring check (never refuses)",
             'if "REPLACE with" in notes:', 'if "REPLACE with" in notes and False:'),
    Mutation("5-template-guard", HANDCALC, HANDCALC_TEST,
             "flip depth-ordering rule (occluder occludes when BEHIND target instead of in front)",
             "if scene.occ_c[2] >= scene.target_c[2]:", "if scene.occ_c[2] <= scene.target_c[2]:"),
    Mutation("5-template-guard", HANDCALC, HANDCALC_TEST,
             "drop the behind-camera (z<=0) refusal check (always False)",
             "if np.any(z <= 0):", "if np.any(z <= 0) and False:"),

    # --- 6. Stub-exclusion + Model D refusal ---
    Mutation("6-model-d", MODELD, MODELD_TEST,
             "invert stub-exclusion condition (is not False -> is not True)",
             'if trial.get("prg_is_stub", None) is not False:\n        return None  # None/True both excluded -- only an EXPLICIT False counts as real',
             'if trial.get("prg_is_stub", None) is not True:\n        return None  # MUTATED'),
    Mutation("6-model-d", MODELD, MODELD_TEST,
             "drop the all-NaN-edges refusal (never returns None for empty real_values)",
             "if not real_values:\n        return None", "if not real_values and False:\n        return None"),
    Mutation("6-model-d", MODELD, MODELD_TEST,
             "zero-real-rows refusal never fires (impossible condition)",
             "if len(real_df) == 0:", "if len(real_df) < 0:"),

    # --- 7. Session scheduler ---
    Mutation("7-session-scheduler", SCHED, SCHED_TEST,
             "rotated_order drops the last object (off-by-one range)",
             "return [objects[(day_index + j) % k] for j in range(k)]",
             "return [objects[(day_index + j) % k] for j in range(k - 1)]"),
    Mutation("7-session-scheduler", SCHED, SCHED_TEST,
             "total_trials uses addition instead of multiplication",
             '"total_trials": trials_per_object_per_day * len(order),',
             '"total_trials": trials_per_object_per_day + len(order),'),
    Mutation("7-session-scheduler", SCHED, SCHED_TEST,
             "flip the Latin-square verification's equality check",
             "return len(set(starts)) == k", "return len(set(starts)) != k"),

    # --- 8. data_quality_report.py ---
    Mutation("8-data-quality-report", DQR, DQR_TEST,
             "confounding check threshold off-by-one (<=1 -> <=0, never flags single-object sessions)",
             "confounded_sessions = per_session_objects[per_session_objects <= 1].index.tolist()",
             "confounded_sessions = per_session_objects[per_session_objects <= 0].index.tolist()"),
    Mutation("8-data-quality-report", DQR, DQR_TEST,
             "prg_stub_vs_real_counts inverts is True check",
             'n_stub = sum(1 for t in trials if t.get("prg_is_stub") is True)',
             'n_stub = sum(1 for t in trials if t.get("prg_is_stub") is not True)'),
    Mutation("8-data-quality-report", DQR, DQR_TEST,
             "confounding_detected flag inverted",
             '"confounding_detected": len(confounded_sessions) > 0,',
             '"confounding_detected": len(confounded_sessions) == 0,'),

    # --- 9. Controller-freeze check ---
    Mutation("9-controller-freeze", DQR, DQR_TEST,
             "off-by-one: allows 2 distinct hashes before flagging",
             "if len(hashes) <= 1:", "if len(hashes) <= 2:"),
    Mutation("9-controller-freeze", DQR, DQR_TEST,
             "NEW_DATASET_TAG check inverted (tag file causes failure instead of allowing)",
             "if tag_path.exists():", "if not tag_path.exists():"),
    Mutation("9-controller-freeze", DQR, DQR_TEST,
             "hard-failure return silently reports ok=True instead of False",
             '"checked": True, "ok": False, "hashes_seen": hashes,\n        "error":',
             '"checked": True, "ok": True, "hashes_seen": hashes,\n        "error":'),

    # --- 10. Depth-unit guard ---
    Mutation("10-depth-unit-guard", DEPTH, DEPTH_TEST,
             "raise the implausibility ceiling 1000x (guard effectively never fires)",
             "MAX_PLAUSIBLE_MEDIAN_DEPTH_M = 20.0", "MAX_PLAUSIBLE_MEDIAN_DEPTH_M = 20000.0"),
    Mutation("10-depth-unit-guard", DEPTH, DEPTH_TEST,
             "mm->m conversion divides by 10 instead of 1000",
             "return depth_mm.astype(np.float32) / 1000.0", "return depth_mm.astype(np.float32) / 10.0"),
    Mutation("10-depth-unit-guard", DEPTH, DEPTH_TEST,
             "zero-valid-pixels guard drops the size==0 check (would crash on median of empty array instead of refusing cleanly)",
             "if valid.size == 0:", "if valid.size == -1:"),
]


if __name__ == "__main__":
    results = run_all(MUTATIONS)
    print_table(results)
