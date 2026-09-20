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

---

## C4 -- Fix the C1 gap estimator, re-run explore/CRITERIA.md unchanged (2026-09-21)

### Root cause (traced via direct code instrumentation before writing
these criteria -- not guessed; see project_closure/LOG.md's C4 entry for
the full instrumented trace)

`_c1_internals()`'s ring is `dilated & ~eroded` -- a band that, by
construction, straddles the bbox border: roughly HALF its pixels sit
strictly inside the original mask's own footprint (before erosion
removes the interior), where `local_gap = |depth - inner_depth| approx 0`
by definition (same flat surface as the interior sample), and the other
half sit outside. Confirmed by direct instrumentation on the idealized
step edge: 40% of all ring pixels have `local_gap == 0` exactly. Taking
the 10th percentile of a population where the "trivially same-surface"
side alone contributes far more than 10% of the ring (true for
essentially any bbox, since the ring is DESIGNED to straddle 50/50) is
mathematically guaranteed to return approximately 0, independent of the
true gap value. This is a different, more precise mechanism than the
xfail-marked test's original docstring guess ("top/bottom ring pixels
dominate") -- the real cause is the straddling ring DEFINITION itself,
not which sides happen to touch the image boundary.

### Fix (fixed before implementing, then implemented, not adjusted after
seeing results)

Change the ring from `dilated & ~eroded` (straddling, ~50% trivially-
same-surface by construction) to `dilated & ~mask` (EXTERIOR-only: the
outward band just past the bbox's own boundary, never including any
pixel that was already inside the original bbox). This removes the
guaranteed same-surface contamination at its source rather than
patching the percentile choice. `cv2.erode()` becomes unused and is
removed. `inner_depth`/`sigma` (computed from the INSET_PX interior
region, unrelated to the ring) are unchanged.

### Acceptance criteria (fixed before implementation)

1. **Root cause converted to a passing test**: the xfail-marked
   `test_c1_gap_estimate_on_clean_step_edge_KNOWN_BUG` becomes a normal
   assertion (`estimated_gap` within a documented tolerance of the true
   1mm gap on the idealized single-edge scene), test renamed to drop
   `_KNOWN_BUG`.
2. **Monotonic response to blur AND noise**, on the same idealized
   step-edge construction already in `tests/test_geometric_prg_error_sanity.py`
   (`|rho| > 0.6`, direction unconstrained per that file's own established
   convention -- physically the direction isn't fixed a priori).
3. **Two-object touching scene**: target bbox directly adjacent to a
   second object (small, ~1mm true gap) with FAR background elsewhere in
   frame (a harder case than the idealized 2-region edge, since
   background contamination -- the E3-diagnosed, separate mechanism --
   is present here too). Estimator's `estimated_gap` is closer to the
   true ~1mm gap than to the far-background distance, i.e. the LOW
   percentile correctly favors the near/touching side over the inflated
   background sides.
4. **Separated scene with a real (non-touching) gap**: target and
   neighbor at a documented larger true gap (e.g. 5cm), far background
   elsewhere. `estimated_gap` recovers a value close to the true 5cm gap
   (not near-zero, not near the background distance).
5. **Background-only scene** (no neighbor at all, only far background on
   every side): `estimated_gap` is large (near the background distance),
   and the final `[0,1]` score reflects "decisively not touching"
   (`score` near 0, since `p_contact` is near 0 for a large gap) --
   reported, not hard-gated to an exact number, since the exact score
   depends on `sigma_gap` too.
6. **Legacy classifier-mode output unchanged**: this fix touches ONLY
   `explore/scripts/geometric_prg_error.py` (the classifier-FREE,
   off-by-default C1/C2/C3 candidates) -- `relation_classifier/scripts/compute_prg_edge_error.py`
   (the classifier-based, default `prg_edge_error_mode`) is untouched.
   `tests/test_prg_edge_error_mode_regression.py`'s existing bit-for-bit
   regression tests must still pass unmodified, proving the default
   mode's output is bit-for-bit identical to before this fix.
7. **Re-run `explore/CRITERIA.md` UNCHANGED**: after the fix, re-evaluate
   C1/C2/C3 against `explore/CRITERIA.md`'s already-pre-committed
   criteria (a)-(e) (non-constant/range/deterministic, sensitivity
   rho>0.6 on >=4/5 worlds, validity AUROC>0.70/CI>0.55, beats baselines,
   collinearity reporting) -- NOT rewritten, NOT loosened, on the SAME 5
   non-held-out worlds `explore/E3_FINDINGS.md` used.
   `world6_dense_mixed` (the held-out world) is touched AT MOST ONCE,
   only if a candidate now passes (a) AND (b) on the 5 non-held-out
   worlds, exactly matching E3's own selection-then-held-out-touch
   protocol -- if none passes, held-out stays untouched, same as E3.
8. **Before/after table reported plainly**, and a direct plain-language
   statement of whether the earlier negative result (E3_FINDINGS.md)
   changes -- not assumed either way before running.

If the Gazebo-based world sweep this criterion needs cannot be run
within the session's remaining budget (headless `gz sim` availability
confirmed in C0, but a full 5-world x noise/blur sweep is a
substantially bigger compute/time cost than the idealized-edge tests
above), this is logged as a blocker with the exact command attempted,
and criteria 1-6 (which don't require the full world sweep) still stand
as this task's minimum bar.

---

## C3 -- YOLO pipeline full verification, software only (2026-09-21)

Scope confirmed by reading `perception-calibration-gap/yolo/` in full
first: `train_yolo_scaffold.py` (mechanics scaffold, network-aware
fallback, EXISTS substantially) and `per_object_gate.py` (per-object
acceptance gate, EXISTS, 6/6 tests from the prior session, but
thresholds are module-level constants, not config -- a real gap against
this task's explicit "(e)" requirement). No auto-labeler exists anywhere
in the repo (confirmed: `grep -rl "auto.label"` only matches
`detector_validation.py`, an unrelated open-vocab detector validator, and
`train_yolo_scaffold.py`'s own docstring mentioning the CONCEPT, not an
implementation) -- MISSING, built fresh. No grouped train/val split
exists (`generate_synthetic_yolo_dataset`'s split is `i < n_train`,
sequential by image index, no session/scene concept at all) -- PARTIAL,
extended.

### Acceptance criteria (fixed before implementation)

**(a) Label round trip**: a new `auto_label_from_markers.py` computes a
YOLO-format label (class_id + normalized xc/yc/w/h) from a detected
ArUco marker's corners, reusing the SAME dilation convention
`capture_and_gap.py`'s `build_bbox_mask_from_corners` already uses (train/
serve parity: the placeholder-box convention must be identical wherever
a marker stands in for a real detector box, not reinvented per script).
A test generates synthetic images (real `cv2.aruco.generateImageMarker`,
same technique as C2's fixtures) with a KNOWN box for apple/purse/glass
(marker placed at a known pixel location, box = the SAME dilation formula
applied to the known corners -- so "ground truth" and "auto-labeler
output" are computed via the identical formula, and the assertion is that
decoding the WRITTEN YOLO label file byte-for-byte reproduces that box
within 1px after the normalize/denormalize round trip). A contact sheet
(boxes drawn on images, SYNTHETIC-stamped filename) is saved for visual
inspection.

**(b) Class index mapping, single source of truth**: a new
`class_map.json` (`{object_name: index}`) is the ONLY place the
apple/tennis_ball/purse/glass/remote -> index mapping is defined.
`train_yolo_scaffold.py`'s `STUDY_OBJECT_NAMES`, `auto_label_from_markers.py`'s
class-id assignment, and `per_object_gate.py`'s object-name handling all
either import this file directly or are tested to be consistent with it
-- a test asserts the labeler's class assignment and the dataset yaml's
`names:` list, read independently, are IDENTICAL.

**(c) Grouped split by session/scene**: `generate_synthetic_yolo_dataset`
(or a new grouped variant) assigns each synthetic image to a session/scene
ID (several images per session, mirroring a real capture session), and
splits train/val/held-out BY SESSION, never by image. A test asserts:
zero session-ID overlap between the resulting splits, and that the same
seed produces the identical split across two runs (determinism).

**(d) Real training loop, 2 epochs, CPU, "code check not detector
result"**: an ACTUAL `model.train(...)` call (not mocked, not skipped) on
the grouped-split synthetic dataset, `epochs=2`, small `imgsz` for speed.
Confirms: training loss is available per-epoch and does not increase
monotonically across the 2 epochs (a weak but genuine "the loop is
learning something," not a strict convergence claim over just 2 epochs);
`last.pt`/`best.pt` checkpoints are written and reloadable
(`YOLO(checkpoint_path)` succeeds); a `model.predict()` call on a held-out
image runs and returns a result object without raising. Every print/log
line from this run states plainly that this is a CODE CHECK on synthetic
solid-color-box data, not a detector-quality result -- matching this
session's SYNTHETIC-stamping discipline throughout.

**(e) Per-object acceptance-gate evaluator**: (i) known-answer tests --
a tiny hand-constructed fixture of predicted vs. ground-truth boxes (2-3
per object) where precision/recall/AP50 are computed BY HAND first (in
the test's own docstring/comments, before running the code) and a new
`detection_metrics.py` (IoU matching, threshold 0.5) is asserted to
reproduce those exact hand-computed numbers -- not just fed
already-computed metric dicts as the existing 6 tests do (those stay,
unchanged, as the aggregate-vs-per-object gate-LOGIC tests; this is a
new, complementary layer testing the metric-COMPUTATION itself). (ii)
Already exists and passes: aggregate-pass/one-object-fails still fails
the gate. (iii) **Thresholds moved out of code literals**: a new
`per_object_gate_config.json` (precision/recall/AP50 thresholds, default
values identical to the current `0.5`/`0.5`/`0.5` constants so existing
behavior is unchanged) is read by `per_object_gate.py`; a regression test
confirms the default config reproduces the exact same pass/fail verdicts
as the hardcoded-constants version did.

**Legacy/regression**: `tests/test_per_object_gate.py`'s existing 6 tests
keep passing UNMODIFIED (config-driven thresholds default to the same
values). `train_yolo_scaffold.py`'s existing network-fallback discipline
(try real, log clearly if falling back) is preserved, not removed, even
though network is confirmed available this session (C0) -- the fallback
path stays exercised/tested for whoever runs this in a restricted
environment later.

If the real 2-epoch training run (d) cannot complete within a reasonable
time budget on this machine's CPU, this is logged as a blocker with the
exact command and wall-clock time attempted, and criteria (a)-(c) and
(e) (all fully achievable without a real training run) still stand as
this task's minimum bar.
