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
