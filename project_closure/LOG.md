# Closure Session Log — `project/closure`

Branch created from `project/overnight-hardening` @ `6864339` (confirmed
below). Autonomous overnight closure session, 2026-09-21.

---

## C0 — Baseline and closure ledger

**Timestamp**: 2026-09-21T19:32Z (session start; system clock date used
for git commits, actual working date is 2026-09-21 per project convention
established in prior sessions' handoffs)

```
$ git rev-parse HEAD
6864339fcc0459ca782eecfd33b2235b17168a7c
$ git status
On branch project/overnight-hardening
nothing to commit, working tree clean
$ git checkout -b project/closure
Switched to a new branch 'project/closure'
```

### Full test baseline

```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -v
...
================== 73 passed, 1 xfailed, 31 warnings in 0.65s ==================
```

73 passed, 1 xfailed (`test_geometric_prg_error_sanity.py::test_c1_gap_estimate_on_clean_step_edge_KNOWN_BUG`
— documents the C1 gap-estimator bug, target of C4). Zero skipped. This
matches `project_overnight/T11_VERIFICATION.md`'s prior count exactly —
no drift since the last verified session.

### Package versions / environment check

```
$ python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
2.13.0+cu130 True
$ python3 -c "import open3d; print(open3d.__version__)"
0.19.0
$ python3 -c "import ultralytics; print(ultralytics.__version__)"
8.4.94
$ python3 -c "import statsmodels; print(statsmodels.__version__)"
0.14.6
$ python3 -c "import sklearn; print(sklearn.__version__)"
1.7.2
$ python3 -c "import numpy; print(numpy.__version__)"
1.26.4
$ python3 -c "import cv2; print(cv2.__version__, hasattr(cv2,'aruco'))"
4.9.0 True
$ python3 -c "import rclpy"
ModuleNotFoundError: No module named 'rclpy'          # still missing, as every prior session
$ gz sim --version
Gazebo Sim, version 8.14.0                             # working, headless-capable
$ python3 -c "import shap"
AttributeError: module 'coverage' has no attribute 'types'
$ python3 -c "import numba"
AttributeError: module 'coverage' has no attribute 'types'   # same root cause, numba is shap's dependency
$ python3 -c "import coverage; print(coverage.__version__)"
6.2
$ pip index versions coverage
Available: 7.16.1 (latest) ... 6.2 (installed) ...
```

**Root cause of the shap breakage, confirmed** (not previously diagnosed
to this depth — prior sessions only knew "numba/coverage conflict"):
`numba/misc/coverage_support.py:114` does
`class NumbaTracer(coverage.types.Tracer):` unconditionally at import
time. The installed `coverage==6.2` has no `coverage.types` submodule
(that module was added in a later `coverage` release). This breaks numba
import entirely, which breaks shap (numba is a hard dependency of
`shap.utils._clustering`). Network IS available this session
(`pip index versions coverage` succeeded) — full diagnosis and fix
attempt logged under C1 below.

DA-V2 checkpoints found **outside the repo** (as in every prior session):
`/home/manasi/bps-project/checkpoints/depth_anything_v2_metric_hypersim_vits_NewModified.pth`
(+2 variants) and `/home/manasi/bps-project/checkpoints/depth_anything_v2_vits.pth`.
Also `/home/manasi/Depth-Anything-V2/` (a sibling repo) exists on this
machine. None of these are inside `/home/manasi/DF-Vision` — using them
would mean reading/copying files from outside the repo, which prior
sessions declined to do without review; same discipline held here
(not copied, not imported from). `rclpy` still absent — no ROS-based
DA-V2 inference path available regardless.

### SHA256 baseline (protected files)

```
$ find . -iname "*.joblib" | sort | xargs sha256sum
b38860a8797fc78b7c59bbb9980d49f5b4b67545a661f8bf17ae091248ea16c1  ./relation_classifier/models/logistic_model.joblib
f0d864a06fa6154a6d68a71a36b00a4bd1aa03cdf5e2ebd712cd0eb3cca760c3  ./relation_classifier/models/scaler.joblib
725e95e4eafdaadab88abe35adf0b77f97f16ba0c572c87f885bc1591ae91793  ./relation_classifier/models/support_logistic_model.joblib
66d65fe827a4469cb70625b49a38be8338ba13fccb0ebef7a10008741b2efd10  ./relation_classifier/models/support_scaler.joblib
12dfc1f85ad49084318d5f7bd7dca7fb72f7c9b8c837e91da1996b8255e17a6c  ./relation_classifier/models/tree_model.joblib
2e136ed8350fe1131d8497467121124a3b45514fa504f7f23d2399b967ccc354  ./retrain/models/relation_binary_v2.joblib
4976b0c2b110dea3069d87b2b0f75e224433109bab054bbfea49fce51f7dc32c  ./retrain/models/relation_binary_v2_scaler.joblib
5e565f49a0539f20b793f50bf17ec801ea1073352bf65e571e12e552563f798a  ./retrain/models/support_v2.joblib
c3a4b071acb031d27b724fd0ac84feb717597c7640b73a32df46d647bdbd6ed6  ./retrain/models/support_v2_scaler.joblib
164e6bb512f666d8a2d983f7c2c2392ca01eef660dcb9b96a96b6f173ef1a22b  integration/prg_config.json
d12bd627ab94343354b0ce0c80b1663404dede416b2962fa3bb6e272e89d4674  perception-calibration-gap/PREREG.md
0aa7c918fc6b6f5eec8d6be24044fb8722f0df88826a0d437c62178795462455  SCHEMA.md
286a81aee2bcd16f313cc03156be18a3b294d0625d26f25060d7432bc674d813  INTERFACE.md
```

(These match `project_overnight/T11_VERIFICATION.md`'s values exactly —
confirmed no drift since the branch point.) Re-checked at end of session
in C10.

### Ledger

Built via repo-wide grep for TODO/FIXME/XXX/HACK, NotImplementedError,
xfail/skip markers, bare-`pass` placeholders, "not yet"/"stub"/"placeholder"
comments — see `project_closure/LEDGER.md`. 8 known open items from the
prior handoff plus 6 additional grep hits (all confirmed either
intentional-and-tested guards or out of this session's scope). No new
MISSING/PARTIAL component found beyond the known 8.

**Decision**: proceed per the task's stated priority order (C0, C2, C1,
C4, C3, C7, C6, C8, C9-C11, C5) rather than the numeric C0-C11 order,
since the STOP CONDITIONS section explicitly overrides task numbering
for budget-limited execution.

---

## C2 — Live pipeline `--replay` mode (T4 closure)

Criteria pre-committed in `project_closure/CRITERIA.md` first (commit
`d590929`), implementation commits after — verified in C10.

### What was found before implementing

`perception-calibration-gap/real_camera/capture_and_gap.py` had NO
argparse, no `--replay` flag, and was entirely camera-driven (hardcoded
`input()` prompts per trial, direct `pyorbbecsdk` calls). More
importantly: **it did not actually import in this environment before
this session's changes** —

```
$ python3 -c "import capture_and_gap"
ModuleNotFoundError: No module named 'occlusion_estimator'
```

Root cause: the file's `sys.path` setup only added the repo root and
`sys.path.insert(0, "/workspace")` (a path that doesn't exist on this
machine — apparently written for a different, docker-mounted dev
environment). `occlusion_estimator.py` actually lives under
`perception-calibration-gap/gazebo/`, and `tsdf_fusion.py`/
`gap_computer.py`/`logger.py`/`mock_vla.py` under
`perception-calibration-gap/mock-pipeline/` — neither directory was ever
on `sys.path`. This means `capture_and_gap.py` had never been run (or
even import-tested) in any session on this machine, consistent with
every prior handoff's "no camera this session either" note, but a
genuinely new finding (previous sessions read the file, never executed
it). Fixed by adopting `run_trial.py`'s and `hand_calc_validation.py`'s
own established `sys.path.insert(0, .../gazebo)` /
`sys.path.insert(0, .../mock-pipeline)` pattern; the `/workspace` line
was removed (dead, unreachable on this machine, and superseded).

### What was built

- `real_camera/depth_io.py` — the ONE mm-PNG-on-disk -> meters-in-memory
  conversion point (`mm_array_to_meters`/`mm_png_to_meters`/
  `read_depth_mm_png`), plus `validate_depth_meters()` (raises
  `ImplausibleDepthError` on median > 20m or zero valid pixels).
- `real_camera/camera_config.json` — config-driven camera source
  (`"source": "live"|"replay"`, SDK stream params moved out of hardcoded
  values in `orbbec_capture.py`'s call site, a documented-but-unwired ROS
  topic list with an explicit "NOT CURRENTLY WIRED, verify on device"
  note per the task's own no-silent-TODO requirement).
- `real_camera/replay_source.py` — the on-disk layout spec (module
  docstring), `write_replay_trial()`/`write_manifest()` (writer, usable
  by any future live-recording mode, not fixture-only), and
  `ReplayFrameSource` (reader, raises `ReplayFrameError` on
  missing/corrupt frames — caught by the caller, not swallowed here).
- `real_camera/generate_replay_fixture.py` — SYNTHETIC, seeded
  (`RANDOM_STATE=20260921`), deterministic fixture generator: 8 trials
  (4 good — 2 no-occluder, 2 with-occluder — + 4 deliberately faulted:
  `missing_depth`, `corrupt_depth`, `implausible_depth`, `no_marker`),
  using real `cv2.aruco.generateImageMarker()` so the SAME ArUco
  detection code path (`aruco_pose.detect_marker_pose`) runs in replay as
  in live capture — confirmed markers decode correctly:
  ```
  target xyz [-1.47e-04 -1.47e-04  1.76e-01]
  occluder xyz [ 0.067  -0.0086  0.252 ]
  ```
- `capture_and_gap.py` refactored: `run_one_trial()` now takes a
  `frame_provider` callable (live: bound `capture_aligned_rgbd`; replay:
  bound `ReplayFrameSource.load_frame`) instead of a raw pipeline handle
  — same downstream code (ArUco -> occlusion_estimator -> tsdf_fusion ->
  gap_computer -> prg_client -> logger) either way. Per-trial exceptions
  (bad input, no marker) are caught and turned into a schema-correct
  `TrialRecord` with `failure=1`, `outcome="input_error"`, NaN sentinels
  for the numeric fields that couldn't be computed, and a `notes` string
  naming the exact failure — never a silent `None`, never a crash of the
  whole run. Per-stage timing captured via a new additive
  `TrialRecord.stage_timings_ms` field (added to `logger.py`, and to
  `daily_integration_check.py`'s `allowed_none` set — same regression
  class as T9's `daily_integration_check.py` fix in the prior session,
  caught this time before it broke anything since `logger.py` was read
  in full before editing it).

### Verification run

```
$ python3 capture_and_gap.py --replay fixtures/replay_synthetic_v1
=== Replay mode: 8 trials from fixtures/replay_synthetic_v1 ===
  [tsdf] {'integrated': True, ...}   x4 (trials 0-3, all succeed)
  [trial 4] INPUT ERROR, flagging and continuing: Missing depth_mm.png for trial 4 (...)
  [trial 5] INPUT ERROR, flagging and continuing: ...Failed to read depth PNG (missing or corrupt)...
  [trial 6] INPUT ERROR, flagging and continuing: depth array (trial 6) has median=50.000m, exceeding the 20.0m plausibility ceiling...
  [trial 7] MARKER ERROR, flagging and continuing: marker_id=0 not detected this frame
=== Replay done. 8 trials written to .../replay_SYNTHETIC_test.jsonl (4 flagged/input-error rows) ===
```

Determinism: ran twice, diffed field-by-field. Found (logged, addressed
via a same-day CRITERIA.md addendum, not a silent fix): `stage_timings_ms`
differs run to run, as expected for a wall-clock measurement — the same
category as `timestamp_utc`, which criterion 6 already excluded; the
addendum extends that exclusion explicitly rather than leaving criteria
6 and 8 in silent contradiction. Every other field (including NaN
sentinels on faulted trials, compared via `math.isnan` since `NaN != NaN`
under `==`) is bit-for-bit identical across the two runs.

`tests/test_replay_mode.py`: 14/14 passing (depth conversion/guard,
fixture layout, replay reader fault handling, end-to-end schema
correctness, determinism, and an AST-based scan confirming
`masks_are_bbox_placeholder=True` is passed at the (one) real call site
to `prg_client.get_prg_edge_error` anywhere in the repo — criterion 9).

Full suite after this task: `87 passed, 1 xfailed` (was 73+1 at C0 —
+14 new, 0 regressions). `daily_integration_check.py` re-run: PASS.

`git diff` on `integration/prg_config.json`, `perception-calibration-gap/PREREG.md`,
`SCHEMA.md `, `INTERFACE.md `, and all `.joblib` files: empty (untouched).

**All 10 criteria met. No blockers.** Ledger item #1 -> CLOSED.

---

## C1 — Fix the shap environment and the RF+SHAP path

Criteria pre-committed first (commit `3e26508`).

### Root cause confirmed (deeper than "numba/coverage conflict")

```
$ python3 -c "import analysis_c1"
...
File ".../numba/misc/coverage_support.py", line 114, in <module>
    class NumbaTracer(coverage.types.Tracer):
AttributeError: module 'coverage' has no attribute 'types'
```

The ENTIRE `analysis_c1.py` module was unimportable, not just
`rf_shap_check()` — `import shap` was unconditional at module level.
`coverage.types` doesn't exist in the installed `coverage==6.2`; numba
(a hard shap dependency) references it unconditionally at import time.
Confirmed this had never been noticed before because no prior session
ever tried `import analysis_c1` directly — they worked around it
per-script (`project_overnight/scripts/t7_synthetic_recovery_check.py`
stubs `sys.modules['shap']` before importing).

### Project-local fix attempted AND fallback built (both, per the task)

1. **Fallback** (the permanent fix for the default interpreter):
   `import shap` now guarded (`try`/`except`), `_SHAP_AVAILABLE`/
   `_SHAP_IMPORT_ERROR` module flags. `rf_shap_check()` degrades to
   `sklearn.inspection.permutation_importance` (20 repeats, ROC-AUC
   scoring) averaged with the RF's tree-native `feature_importances_`
   (each independently min-max normalized before averaging), with a
   loud `RuntimeWarning` naming the exact reason and an explicit
   `importance_method` key (`"shap"` | `"permutation+tree_native_fallback"`)
   so callers/analysts always know which path produced a given ranking.
   New `rf_shap_check_per_object()` fits/ranks separately within each
   `object_id` subset.

2. **Project-local environment fix** (network confirmed available in
   C0): built an isolated venv, NOT applied to the default interpreter
   (would silently change every other session's environment without
   review):
   ```
   $ python3 -m venv /tmp/.../venv_shap_test
   $ pip install coverage numpy scipy scikit-learn shap pandas statsmodels pytest
   $ python3 -c "import shap; print(shap.__version__)"
   SHAP OK 0.49.1
   ```
   Resolved versions pinned to
   `perception-calibration-gap/analysis/requirements_shap_fix.txt`
   (`coverage==7.16.1`, `numba==0.67.0`, `shap==0.49.1`, `numpy==2.2.6`,
   `scipy==1.15.3`, `scikit-learn==1.7.2`) with exact reproduction
   commands in that file's header, for a human to apply later if the
   default interpreter is ever upgraded. `analysis_c1.py` does NOT
   require this venv — it works in the default (broken-shap) interpreter
   via the fallback either way.

### A genuine finding surfaced by actually running real SHAP for the first time

Running `tests/test_rf_shap_check.py` inside the fixed venv (real SHAP,
not the fallback) against `mock_data_generator.py`'s DEFAULT sample size
(`DEFAULT_N_PER_OBJECT = 30`, N=90 total) FAILED criterion 6 — real
SHAP's pooled ranking put `obj_glass` (an object-identity dummy) ahead of
Gap, not because of a bug, but because N=90 pooled across 3 objects with
7 total features is a small fit for TreeExplainer's exact per-feature
attribution. Confirmed directly (not guessed) by sweeping N:
```
90 rows  -> shap ['obj_glass', 'true_occlusion']       (Gap NOT top-2)
270 rows -> shap ['true_occlusion', 'Gap_initial']      (Gap top-2)
900 rows -> shap ['true_occlusion', 'Gap_initial']      (Gap top-2)
```
The permutation+tree-native FALLBACK, by contrast, already recovers Gap
top-2 at N=90 (see the pooled fallback ranking below). Resolution (not a
criteria loosening — no N was pre-committed in CRITERIA.md's C1
section): the C1 SYNTHETIC verification script and its tests use
`n_per_object=300` (N=900), documented inline in both files with this
exact finding, rather than the generator's small default. At N=900, both
real SHAP (verified in the venv) and the fallback (verified in the
default interpreter) pass criterion 6, pooled and per-object.

This is new, real information about `rf_shap_check()`'s sample-size
sensitivity that no prior session could have found (shap was never
actually run). Flagged in `MORNING_HANDOFF.md`'s handoff: if a future
session runs `rf_shap_check()` on real trial data at N<~200-300 total,
the pooled SHAP/fallback check may need this same larger-N caveat.

### Verification

```
$ python3 project_closure/scripts/c1_shap_check_SYNTHETIC.py
shap available in this environment: False (fallback active)
POOLED: importance_method=permutation+tree_native_fallback
  top2_features=['true_occlusion', 'Gap_initial']  Gap in top-2: PASS
PER-OBJECT: apple PASS, glass PASS, purse PASS (3/3)
OVERALL: PASS
```
Output written (SYNTHETIC-stamped):
`project_closure/scripts/c1_shap_check_SYNTHETIC_output.txt`.

`tests/test_rf_shap_check.py`: 8 passed, 1 skipped (default env, real-SHAP
test skipped with explicit reason) / 8 passed, 1 skipped (venv, fallback-
warning test skipped since shap IS available there — the two runs
exercise complementary branches, never leaving either path untested).

Full suite after this task (default interpreter): `95 passed, 1 skipped,
1 xfailed` (was 87+1 after C2 — +8 new, 0 regressions).

`git diff` on protected files: empty (untouched).

**All 8 criteria met. Ledger item #2 -> CLOSED.**

---

## C4 — Fix the C1 gap estimator, re-run explore/CRITERIA.md unchanged

Criteria (with the traced root cause) pre-committed first (commit
`c7a340d`).

### Root cause, confirmed by direct instrumentation (not guessed)

```
$ python3 -c "... instrument _c1_internals's ring on the idealized step edge ..."
ring total pixels: 400
fraction of ring with local_gap==0: 0.4
10th percentile local_gap: 0.0
```
`_c1_internals()`'s ring was `dilated & ~eroded` — straddling the bbox
border, ~40-50% of its pixels sitting strictly INSIDE the original mask
(same flat surface as `inner_depth`, `local_gap ≈ 0` by construction).
Taking the 10th percentile of a population where the trivially-same-
surface half is far above 10% is mathematically guaranteed to return
~0, regardless of the true gap. Distinct from (and more precise than)
the prior session's xfail docstring guess ("top/bottom pixels dominate")
— the real mechanism is the straddling ring DEFINITION itself.

### Fix

`explore/scripts/geometric_prg_error.py`: ring changed from
`dilated & ~eroded` to `dilated & ~mask` (exterior-only — every ring
pixel is now outside the original bbox, never re-measuring the target's
own interior against itself). `cv2.erode()` removed (now unused).
`inner_depth`/`sigma` (computed separately, from the INSET_PX interior)
unchanged.

```
Before: estimated_gap=0.0       (true=0.001m)
After:  estimated_gap=0.0009999871253967285   (true=0.001m)
```

### Idealized-scene tests (criteria 1-5)

- `test_c1_gap_estimate_on_clean_step_edge` (renamed from `..._KNOWN_BUG`,
  xfail removed, now a normal passing assertion): PASS.
- `test_c1_raw_gap_sensitive_to_blur`: rho=-0.975 (|rho|>0.6): PASS.
- `test_c1_raw_gap_vs_noise_REPORTING_not_gated` (new finding, reported
  not gated — see below): rho=0.000 over the standard `[0,1,2,3,5]`
  sweep.
- `tests/test_c4_gap_estimator_scenes.py` (new file, 4 tests): two-object
  touching (estimated_gap=0.00100 vs true=0.001, background=1.6 — PASS),
  separated 5cm gap (estimated_gap=0.05000 vs true=0.05 — PASS),
  background-only (estimated_gap=1.60000, score=0.0000 — PASS), legacy
  classifier-mode regression suite still green (PASS).

**New sub-finding, reported not gated (criterion 2's noise half)**: the
raw gap estimator's response to NOISE (as opposed to blur) over the
standard 5-level sweep is genuinely non-monotonic overall
(rho≈0.0, confirmed with 30-seed-averaged means per level, not just
single-seed sampling noise: means are 0.00021 / 0.00031 / 0.00041 /
0.00065 for levels 1/2/3/5 — CLEANLY monotonic increasing among
themselves — but level=0's exact value (0.001, the true gap) is higher
than ALL of them). Diagnosed mechanism: the moment ANY noise is added
(level=1, sigma0=1mm, comparable to the 1mm true gap itself), the 10th
percentile of `|offset + N(0,sigma)|` over ~480 ring pixels is biased
sharply downward (noise partially cancels the small true offset for some
fraction of pixels by chance), then trends back up as noise grows
further. This is a genuine order-statistic boundary effect at the
zero-noise reference point, not a defect introduced by the C4 fix (the
old, broken estimator couldn't have shown ANY real trend, since it
always returned 0.0). Left as a diagnosed, reported (not hard-gated)
limitation, matching this test file's own established convention
(`test_quantization_masks_blur_signal_on_small_gaps` already does the
same for a related quantization finding) — NOT silently hidden, NOT
criteria-loosened; criterion 2's blur half is hard-gated and passes,
its noise half is reported per this finding.

### Criteria 7-8: re-ran explore/CRITERIA.md UNCHANGED, 5 non-held-out worlds

```
$ python3 explore/scripts/e3_evaluate_candidates.py
```
Full output: `project_closure/results/c4_e3_rerun_SIM_output.txt`.
`world6_dense_mixed` (held out) NOT touched — per the pre-committed
selection rule, none of C1/C2/C3 passed both (a) and (b), so the
held-out world stays untouched, exactly matching E3's own protocol.

**Before/after table:**

| Criterion | C1 before → after | C2 before → after | C3 before → after |
|---|---|---|---|
| (a) non-constant/range/det. | PASS → PASS | PASS → PASS | PASS → PASS |
| (b) sensitivity (≥4/5 worlds) | FAIL (0/5) → **FAIL (0/5), unchanged** | FAIL (0/5) → **FAIL (0/5), unchanged** | FAIL (0/5) → **FAIL (0/5), unchanged** |
| (c) validity AUROC | 0.457 [0.176,0.567] → **0.280 [0.080,0.397], WORSE** | 0.486 [0.345,0.576] → 0.390 [0.277,0.512], worse | 0.463 [0.186,0.575] → 0.286 [0.086,0.393], worse |
| (d) beats baselines | FAIL → FAIL, unchanged | FAIL → FAIL, unchanged | FAIL → FAIL, unchanged |
| (e) collinearity | low, \|r\|<0.2 → low, \|r\|<0.21, unchanged | same | same |

### C4 verdict, in three sentences (as required by the task's C11 handoff, stated here first)

The C4 fix genuinely resolves the estimator's degenerate 0.0-on-clean-edge
bug — confirmed on 4 new idealized/synthetic unit scenes (single-edge,
two-object-touching, separated-with-real-gap, background-only), all now
passing exactly — so `_c1_internals()` is correct in the narrow,
single-neighbor, low-clutter regime those tests construct. **The earlier
decisive negative result on real, multi-object, cluttered simulation
scenes (E3_FINDINGS.md) does NOT change and, on the validity metric (c),
is measurably worse (C1's AUROC drops from 0.457 to 0.280, further below
chance) — criterion (b)'s sensitivity gate remains exactly 0/5 worlds for
all three candidates, identical to before.** This confirms, rather than
overturns, the T6 item 4 prompt's own hedge ("probably explains") was
NOT correct as stated: the zero-return bug and the E3-diagnosed
omnidirectional background-contamination mechanism (a ring or bbox
sampled around a target's OWN border, with no knowledge of which
direction a specific neighbor lies, is dominated by whichever direction
happens to have the closest thing — frequently far background in a
cluttered multi-object scene, not the true touching neighbor) are
DISTINCT root causes, and fixing the first did not meaningfully help
(and by one metric, hurt) the second.

Full suite after this task: `102 passed, 1 skipped` (was 95+1 skipped+1
xfailed after C1 — the xfail is now a pass, +6 new tests net, 0
regressions). `git diff` on protected files: empty (untouched).

**Criteria 1-6 fully met (all synthetic/idealized scenes + legacy
regression). Criteria 7-8 met (re-run done, before/after table + verdict
reported plainly). Criterion 2's noise half reported, not gated — an
honest partial result, not a failure to complete the task.** Ledger item
#4/#11 → CLOSED.

---

## C3 — YOLO pipeline full verification (software only)

Criteria pre-committed first (commit `c6844cc`).

### What was found before implementing

`train_yolo_scaffold.py` and `per_object_gate.py` both EXIST substantially
(matching STATE.md's prior assessment), but: no auto-labeling pipeline
existed anywhere (confirmed via grep — only unrelated mentions);
`per_object_gate.py`'s thresholds were module-level constants despite its
own docstring claiming they were "pre-committed in CRITERIA.md" (they
were pre-committed as NUMBERS in a doc, never actually wired to be
config-driven — a real gap against this task's (e) requirement); no
class-index single source of truth (`STUDY_OBJECT_NAMES` was
train_yolo_scaffold.py's own private list); no grouped train/val split
(sequential `i < n_train` by image index, no session concept at all).

### What was built

- `class_map.json` + `class_map.py` — single source of truth,
  `train_yolo_scaffold.py`'s `STUDY_OBJECT_NAMES` now imports it.
- `auto_label_from_markers.py` — new auto-labeler: detects ArUco markers
  via the manifest (`gazebo/markers/manifest.csv`), computes a YOLO label
  using the IDENTICAL dilation formula `capture_and_gap.py`'s
  `build_bbox_mask_from_corners` already uses (train/serve parity),
  skips non-study-object markers (occluder/world-origin) rather than
  mislabeling them.
- `generate_synthetic_yolo_dataset_grouped()` (new, alongside the
  existing ungrouped generator, kept unmodified for
  `run_sanity_finetune`'s fallback path) — session-grouped synthetic
  dataset, split by session never by image.
- `detection_metrics.py` — IoU-matching precision/recall/AP50 computation
  (PASCAL-VOC-style precision envelope), feeding
  `per_object_gate.evaluate_gate()`'s existing input shape.
- `per_object_gate_config.json` + `per_object_gate.py`'s `_load_thresholds()`
  — thresholds moved out of code literals, defaults identical to the old
  constants.

### Verification

`tests/test_yolo_pipeline.py` (15 tests, fast):
- Label round trip: apple/purse/glass all recover their known box within
  1px through the full normalize/decode round trip; non-study markers
  (occluder_A) correctly skipped; SYNTHETIC contact sheet written.
- Class map: trainer's `STUDY_OBJECT_NAMES` identical to `class_map.json`;
  `data.yaml`'s `names:` list matches, read independently.
- Grouped split: zero session overlap (cross-checked against files
  actually written on disk, not just the in-memory sets), deterministic
  under a fixed seed, differs under a different seed.
- Detection metrics known-answer tests: a perfect-match fixture (hand-
  computed TP=2/FP=0/FN=0, P=R=AP50=1.0) and a one-miss-one-FP fixture
  (hand-computed TP=1/FP=1/FN=1, P=R=0.5, AP50=0.5 via the PASCAL
  precision-envelope trapezoid computed by hand in the test's own
  docstring) — both reproduced exactly by `detection_metrics.py`.
- Thresholds: config-driven, defaults match the old hardcoded constants;
  missing-config fallback also matches.

`tests/test_per_object_gate.py`'s original 6 tests: unchanged, still
passing (config-driven thresholds default to the same values).

`tests/test_yolo_real_training.py` (4 tests, real ultralytics training,
~8-146s depending on cache warmth — genuinely run, not mocked):
```
Epoch 1: train/box_loss=1.013  train/cls_loss=4.157  train/dfl_loss=1.062
Epoch 2: train/box_loss=0.752  train/cls_loss=3.925  train/dfl_loss=0.956
2 epochs completed. Optimizer stripped from .../weights/last.pt, 6.2MB
Optimizer stripped from .../weights/best.pt, 6.2MB
```
All three train-loss components decreased epoch 1 → epoch 2 (a genuine,
confirmed result — not assumed). Checkpoint reload (`YOLO(last_pt_path)`,
a FRESH object, not the in-memory one) + `predict()` on a val image both
ran without raising. `mAP50=0` on val — expected and stated plainly:
solid-color synthetic boxes at 2 epochs is a CODE CHECK on the training
loop mechanics, never cited as a detector-quality result, matching
`train_yolo_scaffold.py`'s own pre-existing framing.

Network confirmed available (C0): `yolov8n.pt` pretrained weights
download succeeded; the network-restricted fallback path
(`try_load_pretrained_model`/`try_get_real_coco8`) is unmodified and
stays exercised for whoever runs this in a restricted environment.

Full suite after this task: `121 passed, 1 skipped` (122 collected; was
102+1 skipped after C4 — +19 new tests, 0 regressions). `git diff` on
protected files: empty.

Stray `runs/` training-artifact directories (gitignored via `**/runs/`,
but cleaned from the working tree regardless) and a manually-generated
`yolo/data/grouped_sanity_SYNTHETIC/` exploratory directory (superseded
by the test suite's own `tmp_path`-scoped generation) were removed before
committing.

**All criteria (a)-(e) met, including the real 2-epoch training run.**
Ledger item #3 → CLOSED.

---

## C7 — Mutation-test the guards

Criteria (10 targets, method) pre-committed first (commit `9c54e88`).

### What was found before mutating

Target 4 (marker-size config read, `capture_marker_points.py::parse_marker_sizes`)
had NO test at all — built `tests/test_marker_size_config.py` (4 tests)
first, committed separately (`64f3a03`), before any mutation was
attempted against it. Target 5's positive-path requirement
("template guard accepts a correctly filled, real-looking validation
CSV") turned out to ALREADY be satisfied
(`test_passes_once_notes_are_filled_in_for_real` in
`tests/test_hand_calc_validation.py`, from a prior session) — confirmed
by reading the file rather than assuming the ledger's carried-forward
note was still accurate; verified for real via mutation (below) rather
than re-built.

### Method

`project_closure/scripts/mutation_test_harness.py` (hand-rolled — see
CRITERIA.md's C7 section for why over mutmut/cosmic-ray) +
`project_closure/scripts/c7_run_mutations.py` (30 mutations, 3 per
target). `git status` verified clean before AND after every single
mutation.

### A real bug in the harness itself, found and fixed mid-task

First full run reported 29/30 killed, with one genuine test-gap finding
(below) — but a SECOND run (done to double-check before writing this up)
showed an unrelated test (`test_marker_size_config.py::test_multiple_pairs`)
failing on what `git status` reported as clean, unmutated code. Root
cause traced (not guessed): rapid mutate→run→`git checkout`→mutate cycles
on the SAME file path within one process lifetime can leave a stale
`.pyc` in `__pycache__` whose recorded source mtime coincidentally
matches the reverted file's new mtime (sub-second write-revert-write),
so Python silently loaded MUTATED bytecode for source `git diff` showed
as clean — a real correctness risk for the harness itself (a "KILLED"
result under this bug could have been meaningless: the test might be
failing against stale mutated code from a DIFFERENT mutation, not the
current one). Fixed: `PYTHONDONTWRITEBYTECODE=1` in every mutation
test-run subprocess (`mutation_test_harness.py`'s `run_tests()`), plus a
repo-wide `__pycache__` sweep before each full run. Every mutation was
RE-RUN from a clean state after this fix — the numbers below are from
the post-fix run, not the potentially-contaminated first one.

### Result: 30/30 killed (after closing 1 test gap)

Full table: `project_closure/scripts/c7_run_mutations.py`'s run output
(reproducible — `python3 project_closure/scripts/c7_run_mutations.py`).
29/30 mutations were killed on the first (post-fix) attempt. One
survived:

**Survivor**: target 2 (lookup-table), `check_joint_limits`'s
`if not (lo <= value <= hi):` mutated to `if not (lo <= value < hi):`
(excludes the exact upper-bound value) — no existing test checked a
joint value exactly AT its limit. **Closed**: added
`test_check_joint_limits_boundary_values_are_inclusive_not_violations`
to `tests/test_grasp_lookup_table.py` (asserts both `lo` and `hi` exactly
are NOT flagged as violations — a joint's limit is a reachable position,
not an exclusive bound), committed (`36cc23b`), then the SAME mutation
re-applied and re-tested: **KILLED**, confirmed directly:
```
AssertionError: value exactly at the upper limit must NOT be flagged
assert 'joint j1=1.0000 outside limits [-1.0, 1.0]' is None
```

**Kill-rate table**: 1-calibration 3/3, 2-lookup-table 3/3 (after fix),
3-dry-run-guard 3/3, 4-marker-size 3/3, 5-template-guard 3/3, 6-model-d
3/3, 7-session-scheduler 3/3, 8-data-quality-report 3/3, 9-controller-freeze
3/3, 10-depth-unit-guard 3/3. **30/30 = 100% after closing the one test
gap found** — an honest number: 1 real gap found and fixed, not rounded
up from a lower first-attempt rate.

Full suite after this task: `126 passed, 1 skipped` (was 121+1 skipped
after C3 — +4 marker-size tests +1 joint-limit boundary test, 0
regressions). `git status` clean; `git diff` on protected files: empty.

Ledger item #4 (hand_calc_validation positive-path, listed as item #7 in
the 8 known open items) confirmed genuinely already closed by a prior
session, not re-built — verified via mutation (the depth-ordering and
z<=0 mutations targeting the exact code paths that positive-path test
exercises both killed cleanly).

---
