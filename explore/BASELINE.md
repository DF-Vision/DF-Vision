# E0 — Baseline Snapshot

Branch `explore/geometric-prg-error`, created from `retrain/relation-classifier` @ `1e72398`.

## HEAD
```
$ git rev-parse HEAD
1e72398ae5e6e87fd44b3ddd08dac7c7d2301395
```

## SHA256 of all model/data artifacts (legacy + v2)
```
079901037563e0887b8fbdba9fdf74ecf13b9da64936ddd1b18bc020b6ba0a26  ./relation_classifier/data/training_data_metric_only.csv
b38860a8797fc78b7c59bbb9980d49f5b4b67545a661f8bf17ae091248ea16c1  ./relation_classifier/models/logistic_model.joblib
f0d864a06fa6154a6d68a71a36b00a4bd1aa03cdf5e2ebd712cd0eb3cca760c3  ./relation_classifier/models/scaler.joblib
725e95e4eafdaadab88abe35adf0b77f97f16ba0c572c87f885bc1591ae91793  ./relation_classifier/models/support_logistic_model.joblib
66d65fe827a4469cb70625b49a38be8338ba13fccb0ebef7a10008741b2efd10  ./relation_classifier/models/support_scaler.joblib
12dfc1f85ad49084318d5f7bd7dca7fb72f7c9b8c837e91da1996b8255e17a6c  ./relation_classifier/models/tree_model.joblib
49b0609669dbbae445d5adbade43b2a73cce773fd9a6001d28df4a2155f63e2b  ./retrain/data/relation_dataset_v2_SIM.csv
2e136ed8350fe1131d8497467121124a3b45514fa504f7f23d2399b967ccc354  ./retrain/models/relation_binary_v2.joblib
4976b0c2b110dea3069d87b2b0f75e224433109bab054bbfea49fce51f7dc32c  ./retrain/models/relation_binary_v2_scaler.joblib
5e565f49a0539f20b793f50bf17ec801ea1073352bf65e571e12e552563f798a  ./retrain/models/support_v2.joblib
c3a4b071acb031d27b724fd0ac84feb717597c7640b73a32df46d647bdbd6ed6  ./retrain/models/support_v2_scaler.joblib
```
**These 11 hashes are the integrity check for this session. Re-verified
in E4. None will be overwritten — anything new gets a new name under
`explore/`.**

## git status
Clean at session start.

## pytest baseline
```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --ignore=lerobot_mujoco_sim
7 passed, 1 warning in 0.50s
```
(The 7 passing tests are `tests/test_relation_model_v2_regression.py`,
added two sessions ago — this remains the only pytest-discoverable suite
outside `lerobot_mujoco_sim`, which still can't collect due to
pre-existing missing `mujoco`/`pyarrow`, unrelated.)

## Environment capability re-checks

**Gazebo headless**: working, `gz sim`, version 8.14.0.

**DA-V2 checkpoints**: exist, machine-wide, all OUTSIDE this repo:
```
/home/manasi/bps-project/checkpoints/depth_anything_v2_metric_hypersim_vits_NewModified.pth
/home/manasi/bps-project/checkpoints/depth_anything_v2_metric_hypersim_vits_NewModified1.pth
/home/manasi/bps-project/checkpoints/depth_anything_v2_vits.pth
```
Per the prior session's finding (`retrain/DA_V2_COMPARISON_SIM.md`),
direct (non-ROS) inference via `models/depth_anything_v2/dpt.py` was
**blocked**: that file imports a `dinov2_layers` submodule never
committed to this repo (only `dinov2.py`, `dpt.py`, `__init__.py` exist
under `models/depth_anything_v2/`). Re-checked this session (see below)
— still true, not re-fixed here for the same reason as before (would mean
copying files from a sibling out-of-repo project directory without
review). This directly determines E3(f)'s answer: DA-V2 remains UNTESTED,
not by omission but by a confirmed, specific, external blocker.

```
$ python3 -c "from models.depth_anything_v2.dpt import DepthAnythingV2"
ModuleNotFoundError: No module named 'models.depth_anything_v2.dinov2_layers'
```

**rclpy**: still not installed (`ModuleNotFoundError`), consistent with
every prior session.

## Reproduce this snapshot
```
git rev-parse HEAD
find . -iname "*.joblib" -o -iname "training_data*.csv" -o -iname "relation_dataset_v2_SIM.csv" | sort | grep -v __pycache__ | xargs sha256sum
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --ignore=lerobot_mujoco_sim
gz sim --version
find / -iname "*.pth" 2>/dev/null
python3 -c "import rclpy"
python3 -c "from models.depth_anything_v2.dpt import DepthAnythingV2"
```
