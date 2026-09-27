# F0 — Baseline

Branch `project/final-integration`, created from `project/prereg-correction`
(`fbb0c3f`). Working tree clean at branch creation.

## Full test suite

```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -v
146 passed, 1 skipped, 304 warnings in 54.31s
```
The 1 skip: `test_rf_shap_check.py::test_rf_shap_check_real_shap_path`,
skipped with an explicit reason (`shap unavailable in this environment:
<exact traceback>`) — verified passing separately in an isolated
project-local venv (`project_closure/LOG.md`'s C1 entry). This is the
same count reported at the end of `project/closure`, `project/audit`,
and `project/prereg-correction` — no drift across three consolidation
passes.

## SHA256 of protected files

```
$ find . -iname "*.joblib" | sort | xargs sha256sum
$ sha256sum integration/prg_config.json perception-calibration-gap/PREREG.md "SCHEMA.md " "INTERFACE.md "
```
```
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
Identical to every prior session's baseline back to `project/closure`'s
C0 entry (`project_closure/LOG.md`). `integration/prg_config.json`
content: `{ "use_stub": true }` — unchanged through 8 sessions.

## git status

```
$ git status --porcelain
(empty)
```

## Environment check

```
torch 2.13.0+cu130, CUDA available: True
cv2 4.9.0, aruco: True
sklearn 1.7.2
statsmodels 0.14.6
open3d 0.19.0
ultralytics 8.4.94
shap: BROKEN (AttributeError: module 'coverage' has no attribute 'types') — known, fixed via fallback in analysis_c1.py (C1, project/closure)
rclpy: NOT INSTALLED — no ROS-based real-camera or DA-V2 inference path available
gz sim: Gazebo Sim, version 8.14.0 (working, headless-capable)
```
No change from `project/closure`'s C0 environment check.
