# Overnight Hardening Log — `project/overnight-hardening`

## [T0] Baseline, branch decision, inventory

```
$ git rev-parse HEAD
f5cac32b9f8f0acd01c576f6a0ce66a9fd2c7be0   # explore/geometric-prg-error, before branching
$ git checkout -b project/overnight-hardening
```
Branch decision reasoning and evidence: `project_overnight/STATE.md`
(base = `explore/geometric-prg-error`, confirmed strict superset of all
7 named branches except one trivial README commit on `failure-calibration`).

Full SHA256 baseline:
```
b38860a8797fc78b7c59bbb9980d49f5b4b67545a661f8bf17ae091248ea16c1  relation_classifier/models/logistic_model.joblib
f0d864a06fa6154a6d68a71a36b00a4bd1aa03cdf5e2ebd712cd0eb3cca760c3  relation_classifier/models/scaler.joblib
725e95e4eafdaadab88abe35adf0b77f97f16ba0c572c87f885bc1591ae91793  relation_classifier/models/support_logistic_model.joblib
66d65fe827a4469cb70625b49a38be8338ba13fccb0ebef7a10008741b2efd10  relation_classifier/models/support_scaler.joblib
12dfc1f85ad49084318d5f7bd7dca7fb72f7c9b8c837e91da1996b8255e17a6c  relation_classifier/models/tree_model.joblib
2e136ed8350fe1131d8497467121124a3b45514fa504f7f23d2399b967ccc354  retrain/models/relation_binary_v2.joblib
4976b0c2b110dea3069d87b2b0f75e224433109bab054bbfea49fce51f7dc32c  retrain/models/relation_binary_v2_scaler.joblib
5e565f49a0539f20b793f50bf17ec801ea1073352bf65e571e12e552563f798a  retrain/models/support_v2.joblib
c3a4b071acb031d27b724fd0ac84feb717597c7640b73a32df46d647bdbd6ed6  retrain/models/support_v2_scaler.joblib
164e6bb512f666d8a2d983f7c2c2392ca01eef660dcb9b96a96b6f173ef1a22b  integration/prg_config.json
d12bd627ab94343354b0ce0c80b1663404dede416b2962fa3bb6e272e89d4674  perception-calibration-gap/PREREG.md
0aa7c918fc6b6f5eec8d6be24044fb8722f0df88826a0d437c62178795462455  SCHEMA.md  (trailing-space filename, real artifact)
286a81aee2bcd16f313cc03156be18a3b294d0625d26f25060d7432bc674d813  INTERFACE.md  (trailing-space filename, real artifact)
49b0609669dbbae445d5adbade43b2a73cce773fd9a6001d28df4a2155f63e2b  retrain/data/relation_dataset_v2_SIM.csv
079901037563e0887b8fbdba9fdf74ecf13b9da64936ddd1b18bc020b6ba0a26  relation_classifier/data/training_data_metric_only.csv
82c8cb691cae63c82ed3819902448b3d3cf7425d05d1ce5beccfb114b675d457  relation_classifier/relation_model_config.json
eefe57e92a6cee3e957a721730fc278055ed7e46e18be90b098d171f9d701896  relation_classifier/prg_edge_error_config.json
```

Environment/package check: see `project_overnight/STATE.md` — summary:
torch 2.13.0+cu130 (CUDA available), cv2 4.9.0 (aruco available), sklearn
1.7.2, statsmodels 0.14.6, open3d 0.19.0, ultralytics 8.4.94 all present
and importable. **shap is BROKEN** (`numba`/`coverage` package conflict,
`AttributeError: module 'coverage' has no attribute 'types'`) — newly
discovered, affects T7. rclpy still missing. DA-V2 direct inference still
blocked (`dinov2_layers` submodule, same as two prior sessions).

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --ignore=lerobot_mujoco_sim`
→ 12 passed (baseline for T11 comparison).

Full component inventory (EXISTS/PARTIAL/MISSING with file:line
evidence): `project_overnight/STATE.md`. Key findings carried into task
order: T1 (calibration) and T2 (grasp lookup table) are genuinely
MISSING and are this session's highest-priority builds; T3, T5, T7 are
substantially EXISTS already (harden, do not rebuild); T9 needs a
schema EXTENSION (additive fields) not a rewrite.

Committed: `project_overnight/STATE.md`, this log entry.

## [T1] Camera-to-arm extrinsic calibration — MISSING -> built

Implemented `perception-calibration-gap/calibration/`:
- `compute_extrinsic_calibration.py`: Kabsch/SVD rigid-transform solver,
  det(R)=+1 verified, held-out validation split, degeneracy guard
  (covariance singular-value ratio), units guard (>5m magnitude refused),
  held-out RMS>2cm refuses to save. Outputs YAML `T_base_from_camera_optical`
  (4x4, meters).
- `capture_marker_points.py`: marker-pose helper, REUSES
  `real_hardware_adapter.detect_markers()`/`CameraCalibration` (does not
  reimplement ArUco/solvePnP), writes camera-frame tvecs into
  `point_pairs.csv`.
- `camera_intrinsics_template.yaml`, `point_pairs_template.csv` (written
  on first run) for the hand-capture workflow.

Verified with hand-derived math: `kabsch()` recovers a known 3D rotation
(0.35 rad about Z) + translation to <1e-8 error on noiseless synthetic
points (checked interactively before writing formal tests).

Tests (`tests/test_extrinsic_calibration.py`, 8/8 passing):
```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_extrinsic_calibration.py -q
8 passed in 0.08s
```
Covers: known-transform recovery (noiseless + noisy), too-few-points
refusal, collinear-input refusal, a corrupted-point-in-held-out-set
refusal (50cm injected error, found via seed search to land in the
held-out split), mm-vs-m units guard, proper-rotation (det=+1) check.

No motor/arm code touched — this is pure geometry/math, satisfies hard
rule 1 trivially (no arm-facing code in this task).

## [T2] Grasp lookup-table execution path — MISSING -> built

`perception-calibration-gap/grasp/lookup_table_executor.py` +
`lookup_table_config_template.yaml`. Nearest-bin lookup with a max
distance (0.05m default, config), joint-limit checking, `pose_source`
always populated (`"lookup_table"` / `"lookup_table_failed"`), dry-run
only unless `--i-am-supervising` (never passed this session). No general
IK solver written -- lookup-table only, per instructions.

Tests (`tests/test_grasp_lookup_table.py`, 10/10 passing): known point ->
expected bin/joints; out-of-reach -> IK_FAILURE, confirmed no
extrapolation; joint-limit violation -> IK_FAILURE, confirmed no silent
clamping; refuses non-dry-run without the supervising flag; `MotorInterface`
mock asserted called zero times in dry-run; base `MotorInterface.send_joint_command`
itself raises `NotImplementedError` (confirms no real backend exists to
accidentally call).
```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_grasp_lookup_table.py -q
10 passed in 0.11s
```

## [T3] Checkpoint 1 template-refusal fix — PARTIAL -> closed

`hand_calc_validation.py` already existed and did most of T3 (independent
hand-calc, MAE, scatter plot). Added `refuse_if_still_template()`,
wired into `main()` right after `write_template_if_missing()`. Verified
it correctly refuses on the REAL committed `validation_scenes.csv`
(still template-identical, confirmed via direct run:
`[REFUSED] validation_scenes.csv row 'apple_baseline_noocc' still
contains the template's literal "REPLACE with" marker...`).
`validation_scenes.csv` itself was NOT modified (rule: never fill it in).

Tests (`tests/test_hand_calc_validation.py`, 11/11 passing): AST import-
independence for both `hand_calc_validation.py` and (transitively)
`occlusion_estimator.py`; 6 synthetic-geometry cases with analytically
known visibility (including a hand-worked-out 50%-overlap case, shown in
the test's own docstring); 3 template-guard cases.
```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_hand_calc_validation.py -q
11 passed in 0.08s
```

## [T6] PRG signal: bounded attempt — T6a done, T6b/T6c deferred (logged, not silently dropped)

T6a (sanity test): see commit `b788786`. Found real signal in C2 (once
isolated from a quantization-masking artifact in the shared noise
model), and a genuine, precisely diagnosed NEW bug in C1's core
estimator (returns 0.0 gap on an idealized single-edge, no-background
scene) — a more fundamental issue than the multi-object background-
contamination mechanism `explore/E3_FINDINGS.md` already found. Per rule
7 (max ~3 attempts then log and continue), this bug is recorded
(xfail-marked test, not fixed) rather than patched — fixing it properly
would mean redesigning the ring/percentile logic and re-running the full
pre-committed `explore/CRITERIA.md` (a)-(f) suite again, which this
session's remaining time budget does not support alongside T7-T13.

**T6b (changed-interface neighbor-localization variant) and T6c
(re-evaluation against explore/CRITERIA.md) are explicitly DEFERRED, not
attempted this session.** Building a from-depth-only neighbor
localization (connected components / depth discontinuities) is a
nontrivial new implementation, and re-running the full (a)-(f) gate
suite on it would itself be a multi-hour task given last session's own
timing notes. Given this session's primary new deliverables (T1, T2) and
the remaining T7-T13 scope, this was judged lower priority than breadth
across the rest of the task list. **`use_stub` in `integration/prg_config.json`
remains `true`** — no proposal to flip it is being made (T6d's "if
criteria fail, STOP" branch already applies from last session's
decisive C1/C2/C3 failure; nothing in T6a's new bug findings changes
that verdict, since C1 was already failing decisively before this
session for a different, also-real reason).
