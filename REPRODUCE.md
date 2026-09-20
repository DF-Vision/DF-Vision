# REPRODUCE.md — Skeleton

Output-oriented checklist: which command (see
`perception-calibration-gap/README.md` for the full list) produces which
figure/table this project's paper would cite, and whether it's real or
SYNTHETIC. This is a SKELETON per C9 — fill in real figure/table numbers
once the paper's outline exists; the commands and SYNTHETIC/real
labeling are accurate as of `project/closure` (2026-09-21).

## Person A (Manasi) — hardware, calibration, TSDF/Gap, statistics, conformal gate

| Output | Command | Status |
|---|---|---|
| T1 camera-to-arm extrinsic transform | `python3 perception-calibration-gap/calibration/compute_extrinsic_calibration.py` | Infrastructure ready; **needs real point_pairs.csv from the physical rig** — see `project_overnight/HARDWARE_CHECKLIST.md` |
| T2 grasp lookup-table dry-run | `python3 perception-calibration-gap/grasp/lookup_table_executor.py` | Infrastructure ready, dry-run only, **needs a real hand-teleop-calibrated table** |
| Checkpoint 1 (ground-truth marker validation) | `cd perception-calibration-gap && python3 hand_calc_validation.py` | **Refuses on template data — Checkpoint 1 not yet started on real hardware** |
| Gap-only Model B coefficient table (C1) | `python3 perception-calibration-gap/analysis/analysis_c1.py` | SYNTHETIC (mock data) until real trials exist |
| C1 power curve / underpowering finding | `python3 project_closure/scripts/c6_statistics_SYNTHETIC.py` | SYNTHETIC — **but the underpowering conclusion itself is a real methodological finding, see PREREG_AMENDMENTS_PROPOSED.md item 7** |
| C1 per-object generalization table | `python3 perception-calibration-gap/analysis/analysis_c1.py` (`per_object_descriptive_table`) | SYNTHETIC |
| YOLO per-object detection gate table | `python3 perception-calibration-gap/yolo/per_object_gate.py` | SYNTHETIC demo; **needs real auto-labeled captures per Phase 1.5** |
| Conformal gate coverage table (C4, Gap-only) | `python3 perception-calibration-gap/analysis/analysis_c4.py` | SYNTHETIC |
| Mutation kill-rate table (guards) | `python3 project_closure/scripts/c7_run_mutations.py` | Real (tests the TEST SUITE, not trial data — not "SYNTHETIC" in the trial-evidence sense) |

## Person B (Krupa) — PRG construction, relation classifier, boundary/relation features

| Output | Command | Status |
|---|---|---|
| Legacy binary touching-classifier behavior | (see `relation_classifier/scripts/classify_relation.py`) | Decisively negative, 4 sessions of history — `PREREG_AMENDMENTS_PROPOSED.md` item 5/6 |
| Classifier-free `prg_edge_error` candidates (C1/C2/C3) | `python3 explore/scripts/e3_evaluate_candidates.py` | SYNTHETIC/SIM; decisive negative result, unchanged (and slightly worse on one metric) after this session's C4 estimator fix |
| Structural independence audit | (`explore/scripts/check_structural_independence.py`) | Confirmed, both directions |
| Model D (two-signal, guarded) | `python3 perception-calibration-gap/analysis/model_d.py` | Refuses by default; **conditional on a working PRG signal, which does not currently exist** |

## Not yet producible by anything in this repo

- Real detector (YOLO) quality on the actual study objects.
- Real-depth (DA-V2 or Orbbec) behavior of any PRG signal candidate —
  everything above is simulation/ray-cast only.
- Any failure-prediction claim — needs real grasp trials.
- Calibration residual on real measured points.

See `project_closure/MORNING_HANDOFF.md`'s "What software CANNOT tell
you until real data exists" section for the full, current list.
