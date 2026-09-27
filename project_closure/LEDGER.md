# Closure Ledger — `project/closure`

Built 2026-09-21 per C0. Repo-wide grep for TODO/FIXME/XXX/HACK,
NotImplementedError, xfail/skip markers, bare `pass` placeholders, and
"not yet"/"stub"/"placeholder" comments, scoped to the actual project
(`perception-calibration-gap/`, `relation_classifier/`, `analysis/`,
`integration/`, `tests/`, top-level docs). `lerobot_mujoco_sim/scripts/*`
and `scripts/*` (arm-sim/legacy infra, not part of this project's
T1-T9/C0-C11 scope) are noted but not tracked as rows — their `pass`
statements are exception-handler no-ops (`except Exception: pass` after a
best-effort cleanup comment), not placeholder function bodies, and none
of that code is touched by anything this session builds.

Status values: **CLOSED** (evidence pointer given) or **BLOCKED** (exact
reason given). Every row below must end this session in one of those two
states — see final status column, updated as each task lands.

## The 8 known open items from the previous run

| # | Item | Severity | Plan | Final status |
|---|---|---|---|---|
| 1 | T4: `--replay` mode not implemented (`perception-calibration-gap/real_camera/capture_and_gap.py` is camera-only, hardcoded `input()` prompts, no argparse at all) | blocks-real-data | C2: add `--replay <dir>` entry point, config-driven camera source | CLOSED — [C2](#c2) |
| 2 | `rf_shap_check()` broken: `shap` import fails (`AttributeError: module 'coverage' has no attribute 'types'`, numba/coverage version conflict) | blocks-analysis | C1: diagnose root cause, fix project-locally or fallback | CLOSED — [C1](#c1) |
| 3 | YOLO training loop (`train_yolo_scaffold.py`) never verified beyond synthetic smoke test (previously timed out) | blocks-analysis | C3: full verification incl. real 2-epoch training run | CLOSED — [C3](#c3) |
| 4 | T6: C1 gap estimator (`explore/scripts/geometric_prg_error.py::_c1_internals`) returns 0.0 on idealized single-edge scene, xfail-documented not fixed | blocks-analysis | C4: fix root cause (diagnosed: whole-perimeter ring dominated by non-edge sides), convert xfail to pass, re-run explore/CRITERIA.md | CLOSED — [C4](#c4) |
| 5 | T6b/T6c (changed-interface `prg_edge_error` attempt, full gate re-eval) not done | blocks-analysis | C5: bounded attempt behind config switch, after C4 | CLOSED — [C5](#c5): built, tested on 4 idealized scenes, evaluated against explore/CRITERIA.md, decisive negative (fails criterion b, 0/5 worlds), not selected, held-out world untouched |
| 6 | Null synthetic generator fixed (was zeroing Gap's variance); 5% FPR figure needs re-establishing on more datasets | blocks-analysis | C6a: 1000+ null datasets, exact binomial CI | CLOSED — [C6](#c6) |
| 7 | `hand_calc_validation.py` template guard added; only refusal path tested, not positive (accept) path | blocks-analysis | C7: add positive-path test (real-looking filled CSV) as part of mutation testing pass | CLOSED — [C7](#c7) |
| 8 | Commits on other branches not on this one, listed in STATE.md as "needs a human merge decision" | N/A (explicitly out of scope) | Not to be merged by this session, per task instructions | CLOSED — left untouched, confirmed via `git branch -a` (no merges performed this session) |

## Additional ledger rows found by grep (beyond the 8 known items)

| # | File:line | What's missing | Severity | Plan | Final status |
|---|---|---|---|---|---|
| 9 | `perception-calibration-gap/mock-pipeline/detector_validation.py:157` | `run_detector_on_scene()` raises `NotImplementedError` when no `detector_fn` injected — no real open-vocabulary detector wired in. By design (docstring: "Do NOT stub this with a fabricated bbox") | cosmetic (intentional guard, already tested via injected `detector_fn` in its own test suite, out of scope for T5/C3 which is about the closed-vocabulary YOLO scaffold, not this open-vocab path) | No action — correct as-is | CLOSED — verified intentional via docstring; not part of C3's YOLO scope |
| 10 | `perception-calibration-gap/grasp/lookup_table_executor.py:52` | Base-class `MotorInterface.send_joint_command()` raises `NotImplementedError` | cosmetic (abstract base, tested by `tests/test_grasp_lookup_table.py::test_base_class_motor_interface_raises_not_implemented`) | No action | CLOSED — already correct + tested |
| 11 | `tests/test_geometric_prg_error_sanity.py:195` | `pytest.xfail(...)` documenting the C1 gap-estimator bug (item #4 above) | blocks-analysis | Same as #4 — C4 converts this to a passing assertion | CLOSED — [C4](#c4) |
| 12 | `scripts/grasp_estimator.py:634` | `TODO: this voxel-density + radius-threshold heuristic picks a ...` (heuristic grasp-point picker, arm-sim legacy code, not part of this project's Gap/PRG/stats scope) | cosmetic, out of scope | No action — outside C0-C11 scope (arm-sim heuristic, not touched by any task here) | CLOSED — out of scope, confirmed by file path (not under `perception-calibration-gap/`, `relation_classifier/`, or `analysis/`) |
| 13 | `perception-calibration-gap/run_trial.py:177,199,369` | Three `TODO` comments: real per-object noise model, a documented-partial relationship, and "real-physics session-baseline protocol not yet defined" | cosmetic (this file drives the MuJoCo mock pipeline, explicitly synthetic-only per its own docstring; not part of any C0-C11 task's required deliverable) | No action this session — flagged for a human, would be new scope beyond the 11 tasks | BLOCKED — out of the C0-C11 task list; flagged in handoff for future work, not silently dropped |
| 14 | `scripts/send_grasp_to_arm.py:22` | `TODO (tomorrow, Phase 2 of the hardware pipeline doc)` | cosmetic, out of scope (real-arm script, HARD RULE 1 forbids any arm-motion work this session regardless) | No action | CLOSED — out of scope by hard rule |

Rows 9-14 confirm no additional MISSING/PARTIAL items inside this
project's actual scope beyond the 8 already known — the grep was
comprehensive (TODO/FIXME/XXX/HACK, NotImplementedError, xfail/skip,
bare-pass, "not yet"/"stub"/"placeholder") and every hit outside the 8
known items was either an intentional, already-tested guard or genuinely
outside this session's 11-task scope.

## Status legend for section headers below
Populated incrementally as each C-task lands; see `project_closure/LOG.md`
for full command/output evidence and `MORNING_HANDOFF.md` for the final
rollup.
