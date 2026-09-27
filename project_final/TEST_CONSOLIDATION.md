# F4 — Full-Repo Test Consolidation

## Where the tests live

Since every branch in this project's chain (`unified-pipeline` through
`project/prereg-correction`) is a linear ancestor of the current branch
(confirmed in `project_final/BASELINE.md`'s branch-ancestry check), there
is no scattered, un-merged `tests/` directory from another branch to pull
in — everything is already present in this working tree. The scattering
that exists is a small number of standalone verification SCRIPTS living
outside `tests/`, catalogued below.

## The `tests/` pytest suite (146 passed, 1 skipped)

```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q
146 passed, 1 skipped, 304 warnings in 53.12s
```

**The 1 skip, justified by name:**
`tests/test_rf_shap_check.py::test_rf_shap_check_real_shap_path` —
skipped with an explicit, non-silent reason string
(`f"shap unavailable in this environment: {c1._SHAP_IMPORT_ERROR}"`),
because this environment's `shap` install is broken (root-caused in
`project/closure` C1: a `coverage`/`numba` version conflict). The
fallback path this skip routes around IS tested and passing
(`test_rf_shap_check_warns_loudly_when_shap_unavailable`, etc.); the real
SHAP path was independently verified working in an isolated
project-local venv (`project_closure/LOG.md`'s C1 entry) — this skip is
not silently dropped coverage, it's a documented environment limitation
with a verified-elsewhere fallback.

No duplicate or superseded test files were found inside `tests/` itself
requiring removal — every file there was built incrementally across
`project/closure`'s C-tasks with no earlier-branch equivalent to diff
against.

## Standalone `test_*.py` files OUTSIDE `tests/` (not part of the 146/1 count)

```
$ find . -iname "test_*.py" -o -iname "*_test.py" | grep -v __pycache__ | grep -v "^\./tests/"
./lerobot_mujoco_sim/run_mujoco_grasp_smoke_test.py
./lerobot_mujoco_sim/scripts/tests/test_parquet.py
./perception-calibration-gap/mock-pipeline/test_prg_stub_integration.py
./perception-calibration-gap/test_real_adapter.py
./retrain/scripts/r5_final_test.py
```

**2 are out of scope** (`lerobot_mujoco_sim/*`) — arm-simulation
infrastructure (MuJoCo grasp pipeline, lerobot parquet dataset format),
unrelated to this project's C1-C5/PRG/power-analysis scope, not touched.

**1 is a historical, one-time-use script, correctly NOT re-run**
(`retrain/scripts/r5_final_test.py`) — the R5 held-out-world evaluation,
already executed exactly once (per this project's own "held-out data
touched once" discipline) with its result already recorded in
`retrain/GATES_RESULTS.md` and folded into
`project_final/PRG_SIGNAL_STATUS.md`'s chronological history (item #2).
Re-running it would violate the same held-out-data discipline this whole
project enforces everywhere else — not run.

**2 are in-scope, standalone verification scripts, both run this session:**

1. `perception-calibration-gap/mock-pipeline/test_prg_stub_integration.py`
   — run from its own directory (`cd perception-calibration-gap/mock-pipeline
   && python3 test_prg_stub_integration.py`): **PASSED**
   ("STUB INTEGRATION TEST PASSED", exit 0). Confirms `compute_prg_edge_error()`
   wires correctly into `TrialRecord`/`TrialLogger`'s real schema on
   synthetic test masks. (Note: running it from the repo root instead
   creates a stray `data/` directory via a CWD-relative path — same class
   of issue documented for `mock_data_generator.py`/`hand_calc_validation.py`
   in `perception-calibration-gap/README.md`'s C9 note; not fixed here,
   out of this task's scope, flagged for awareness.)

2. `perception-calibration-gap/test_real_adapter.py` — run: **FAILED**
   ("SOME TESTS FAILED — do not wire this into run_trial.py yet").
   **Root-caused this session (2026-09-27), not silenced:**

   The script's own `synthesize_marker_frame()` helper renders a
   synthetic ArUco frame by projecting `_marker_object_points()`'s 4
   corners with `cv2.projectPoints`, then warping the marker bitmap onto
   those projected corners with `cv2.getPerspectiveTransform`. Direct
   instrumentation (this session) confirmed: swapping the SOURCE
   bitmap's corner Y-ordering (top-row vs. bottom-row) changes detection
   from `ids=None, rejected=4` to `ids=[[0]], rejected=3` — an exact,
   reproduced confirmation that `_marker_object_points()`'s object-frame
   Y-axis convention does not match the raster image's Y-axis convention
   the way `synthesize_marker_frame()` assumes, so EVERY frame this
   script generates is silently Y-mirrored, which corrupts the ArUco bit
   pattern (the detector still finds valid-shaped quadrilateral
   candidates — confirmed via `rejected` count > 0 — they just fail
   dictionary bit-decoding).

   **This is a bug in the TEST SCRIPT's own frame-synthesis helper, not
   in `real_hardware_adapter.py`'s actual `detect_markers()`/
   `rvec_tvec_to_pose()` code under test.** That code is exercised via a
   DIFFERENT, working synthesis method elsewhere in this project
   (`perception-calibration-gap/real_camera/generate_replay_fixture.py`,
   built in `project/closure` C2, which pastes `generateImageMarker()`'s
   bitmap directly into a pixel region rather than perspective-warping
   it) and has been repeatedly confirmed correct there (14/14 tests,
   `tests/test_replay_mode.py`).

   **Not fixed** — fixing this script's corner-ordering convention is
   new script-authoring, out of this consolidation session's scope (it
   would not merge or verify existing work; it would write new code).
   Documented here, dated, with the exact evidence, per this task's
   "root-caused, not silenced, not skip-marked without an explicit dated
   reason" instruction. Recommend a future session fix
   `synthesize_marker_frame()`'s corner ordering if this script's real
   coverage (independent-of-adapter pose-recovery validation) is wanted
   again — `real_hardware_adapter.py`'s actual marker detection is
   already otherwise verified via the working `generate_replay_fixture.py`
   path.

## Final counts

`tests/` suite: **146 passed, 1 skipped** (justified above), **0
failed**. Standalone scripts outside `tests/`: 2 out of scope (untouched),
1 historical/correctly-not-rerun, 2 run this session (1 pass, 1 fail —
root-caused as a pre-existing test-script bug, not a code regression).
