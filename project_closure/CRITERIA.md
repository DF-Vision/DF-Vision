# Closure Criteria — pre-committed, per task

Written BEFORE that task's implementation, in commit order (verified by
`git log` in C10). Never loosened after seeing results. This file is
append-only within a session: each task gets its own dated section below,
committed alone before any of that task's implementation commits.

---

## C2 — Live pipeline `--replay` mode (2026-09-21)

**Scope**: `perception-calibration-gap/real_camera/capture_and_gap.py`
(the only real-camera trial entry point in the repo — confirmed via
`grep -rl "def run_one_trial\|def main" perception-calibration-gap/real_camera/`).

### Acceptance criteria (fixed before implementation)

1. **`--replay <dir>` flag exists** on `capture_and_gap.py`'s CLI and runs
   the exact same per-trial code path as live capture (ArUco pose ->
   `occlusion_estimator` -> `tsdf_fusion` -> `gap_computer` ->
   `prg_client` -> `logger`) with frames read from disk instead of the
   Orbbec SDK. No camera, no `pyorbbecsdk` import required to run replay
   mode (verified: replay mode importable/runnable in an environment
   where `pyorbbecsdk` is absent or unmocked).
2. **Camera source is config-driven**: a `camera_config.json` (or
   equivalent) declares `source: "live" | "replay"` and SDK stream
   parameters; no resolution/format/topic value is hardcoded inside
   `capture_and_gap.py` itself for the parts that vary by source. A
   documented expected-topic list is present (even though this repo's
   live path uses the pyorbbecsdk `Pipeline`/`Config` API directly, not
   ROS topics — the config explicitly states this and lists the ROS
   topic names to verify IF a ROS-based Orbbec driver is substituted
   later, satisfying the task's "Orbbec Gemini 2 topic/SDK names in
   config" requirement without inventing a ROS dependency this repo
   doesn't have).
3. **Depth units**: exactly ONE function performs the on-disk-mm-PNG ->
   in-memory-meters conversion, used by both the replay reader and
   (where applicable) any future recorder. A test asserts a synthetic
   millimeter-valued uint16 PNG converts to the correct meter values. A
   guard function rejects (raises, with the array's median value in the
   message) any depth array whose median exceeds 20.0 (a value that
   cannot be a real meters-scale depth for this project's benchtop
   scenes, and is the task's own stated example threshold) — this guard
   is a separate, directly-tested unit, not folded silently into the
   conversion function.
4. **Fixture set**: `perception-calibration-gap/real_camera/generate_replay_fixture.py`
   produces a SYNTHETIC (filename-stamped `SYNTHETIC`), deterministic
   (fixed seed, documented) set of on-disk frames in the SAME layout
   `capture_and_gap.py`'s replay reader consumes — i.e., the reader and
   generator share one documented layout spec, not two independently
   guessed ones.
5. **Schema-correct output**: one `--replay <fixture_dir>` invocation
   over the fixture set produces `TrialRecord`-shaped JSONL rows (all
   `REQUIRED_FIELDS` present) via the existing `TrialLogger`.
6. **Bit-for-bit determinism**: two separate replay runs over the same
   fixture directory produce identical output, field-for-field, EXCEPT
   `timestamp_utc` (which is wall-clock by design, per `logger.py`'s own
   `field(default_factory=...)` — excluding it from the comparison is not
   a loosening of "deterministic," it is the pre-existing, unrelated
   behavior of every trial record in this codebase).
7. **Corrupt/missing input handled without crashing**: a fixture
   trial with a missing depth file, a corrupt (truncated/unreadable) PNG,
   an all-zero mask, and an implausible (all-`NaN`, or median > 20m)
   depth array are each handled by: (a) a logged warning identifying the
   trial and the specific failure, (b) a row-level flag in the emitted
   `TrialRecord.notes` field (not a silent `None`), (c) the replay run
   continuing to the next trial rather than raising out of `main()`. A
   test constructs one fixture directory containing exactly one bad
   trial of each kind among otherwise-good trials and asserts: all good
   trials are logged correctly, all bad trials are logged with a
   `notes` flag (not silently dropped, not crashing the run), and the
   process exits 0.
8. **Per-stage timing logged**: each trial's per-stage wall-clock time
   (capture/load, marker pose, TSDF, Gap, PRG, logger append) is
   captured in a new additive `TrialRecord.stage_timings_ms: Optional[dict]`
   field (default `None`, existing `REQUIRED_FIELDS` and every existing
   caller/test unaffected — same discipline as every prior additive
   `TrialRecord` extension in this codebase) and printed per trial.
9. **`masks_are_bbox_placeholder` verified at every call site**: a test
   enumerates every call to `integration.prg_client.get_prg_edge_error`
   (or its wrapped variants) anywhere in the repo (via `grep`/AST, not a
   hand-maintained list that can silently go stale) and asserts each one
   passes `masks_are_bbox_placeholder=True` given that neither the live
   nor replay path in this repo has a real segmentation model — the
   masks are always marker-corner-bbox-derived.
10. **Legacy behavior unchanged**: the pre-existing live-capture path
    (`main()` with no `--replay` flag) is behaviorally unchanged —
    verified by a diff-level read of the refactored `run_one_trial`
    showing the live branch's argument values match what the
    pre-refactor code passed, and by keeping the live path's own
    `input()`-per-trial prompts and 8-trial structure exactly as before.

None of the above is loosened after implementation. If any criterion
can't be met within the 3-attempt budget (rule 6), it is logged as a
blocker in `project_closure/LOG.md`'s C2 entry, not silently dropped.

### Addendum, same day, written after first implementation run (logged, not silent)

Criterion 6 ("identical... EXCEPT `timestamp_utc`") and criterion 8
(per-trial wall-clock stage timing must be captured) are in tension:
`stage_timings_ms` is, by criterion 8's own requirement, wall-clock
measured and therefore cannot be bit-identical across runs, exactly like
`timestamp_utc` already wasn't. This was an oversight in how criterion 6
was scoped, not a substantive result being loosened -- the underlying
scientific/schema content (every field except the two wall-clock-derived
ones) is still required to be, and was verified to be, bit-for-bit
identical. **Determinism check is therefore: identical field-for-field
EXCEPT `timestamp_utc` and `stage_timings_ms`.** This is a scope
clarification of an internal inconsistency between two criteria written
in the same commit, not a pass/fail bar being relaxed after a failing
result -- no criterion's actual pass/fail THRESHOLD changed.

---

## C1 -- Fix the shap environment and the RF+SHAP path (2026-09-21)

### Diagnosis (done before writing these criteria, since a fix can't be
scoped without knowing the mechanism -- this is diagnosis, not
implementation)

`import shap` fails at MODULE level in `analysis_c1.py`, meaning the
ENTIRE module (not just `rf_shap_check()`) is unimportable in this
environment -- confirmed: `python3 -c "import analysis_c1"` raises the
same `AttributeError: module 'coverage' has no attribute 'types'`
traceback as a bare `import shap`/`import numba`. Root cause (C0):
`numba/misc/coverage_support.py` references `coverage.types.Tracer`
unconditionally at import time; installed `coverage==6.2` predates that
submodule. `shap` hard-depends on `numba`. Network is available this
session (confirmed in C0); a version-pinned local fix is attempted
first, with the fallback as the acceptance path regardless of whether
the local fix succeeds, since a fix that only works in a throwaway venv
doesn't help the DEFAULT interpreter every other test in this repo runs
under.

### Acceptance criteria (fixed before implementation)

1. **Root cause stated precisely** (done above) with the exact import
   error text, not just "numba/scipy conflict."
2. **`analysis_c1.py` is importable regardless of whether `shap` is
   importable** -- `import shap` is wrapped so a broken shap install
   degrades the module to fallback mode, not an unimportable file. A
   test asserts `import analysis_c1` succeeds in THIS environment
   (where shap is currently broken).
3. **Fallback, same function signature**: when `shap` is unavailable,
   `rf_shap_check()` returns the SAME dict shape (`rf_cv_auc_mean`,
   `rf_cv_auc_std`, `shap_importance_ranked`, `top2_features`) computed
   from `sklearn.inspection.permutation_importance` PLUS the RF's own
   tree-native (Gini) `feature_importances_` (averaged, both reported
   individually too) -- not silently returning a different shape a caller
   would have to special-case. A loud, explicit warning is printed/logged
   naming that SHAP is unavailable and which fallback is in use. An
   added `importance_method` key states `"shap"` or
   `"permutation+tree_native_fallback"` explicitly -- never silently
   ambiguous about which path produced a given ranking.
4. **A test exercises the real SHAP path when importable**, guarded by
   `@pytest.mark.skipif(not _SHAP_AVAILABLE, reason=<the exact import
   error text>)` -- never a silent/bare skip with no reason, per the
   task's explicit requirement.
5. **Pooled AND per-object**: `rf_shap_check()` (pooled, `object_id`
   one-hot as already implemented) unchanged in its pooled behavior; a
   new `rf_shap_check_per_object()` fits/ranks importance separately
   within each `object_id` subset (no cross-object dummy columns inside
   a single-object subset -- those would be constant and uninformative)
   and returns a dict keyed by `object_id`.
6. **Acceptance on synthetic planted data**: using
   `mock_data_generator.py`'s planted-effect output (`TRUE_COEFS['b_gap']
   = 2.5`, an intentionally strong, known signal), the planted Gap
   feature (`Gap_initial` or `Gap_max`) ranks in the top-2 features by
   whichever method (`shap` or fallback) is active in this environment --
   both pooled and for at least one individual object with enough
   trials to fit.
7. **SYNTHETIC-stamped output**: any table/plot this task's verification
   script produces is written under a filename containing `SYNTHETIC`
   and states "SYNTHETIC - NOT A RESULT" in its own header, per hard
   rule 2 -- same discipline as every other synthetic artifact in this
   session.
8. **Legacy behavior unchanged**: every existing caller of
   `rf_shap_check()` (there is exactly one, `analysis_c1.py`'s own
   `if __name__ == "__main__"` block) keeps working with no changes to
   its call site -- the fallback is an internal behavior change only,
   not an interface change.

If the local environment fix (coverage/numba upgrade) cannot be applied
project-locally without a working project-local install path within the
3-attempt budget, this is logged as a blocker and the fallback path
becomes the acceptance path in THIS environment permanently, per rule 10
("no network assumptions... continue with the documented fallback").
