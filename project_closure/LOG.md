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
