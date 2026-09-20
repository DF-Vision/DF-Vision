# R0 — Baseline Snapshot

Branch `retrain/relation-classifier`, created from `fix/boundary-features` @ `44a4979`.

## HEAD

```
$ git rev-parse HEAD
44a4979463d6681adb416ab1018e81aacdef4294
```

## SHA256 of every legacy .joblib/scaler/training CSV

```
$ find . -iname "*.joblib" -o -iname "training_data*.csv" | sort | grep -v __pycache__ | xargs sha256sum
079901037563e0887b8fbdba9fdf74ecf13b9da64936ddd1b18bc020b6ba0a26  ./relation_classifier/data/training_data_metric_only.csv
b38860a8797fc78b7c59bbb9980d49f5b4b67545a661f8bf17ae091248ea16c1  ./relation_classifier/models/logistic_model.joblib
f0d864a06fa6154a6d68a71a36b00a4bd1aa03cdf5e2ebd712cd0eb3cca760c3  ./relation_classifier/models/scaler.joblib
725e95e4eafdaadab88abe35adf0b77f97f16ba0c572c87f885bc1591ae91793  ./relation_classifier/models/support_logistic_model.joblib
66d65fe827a4469cb70625b49a38be8338ba13fccb0ebef7a10008741b2efd10  ./relation_classifier/models/support_scaler.joblib
12dfc1f85ad49084318d5f7bd7dca7fb72f7c9b8c837e91da1996b8255e17a6c  ./relation_classifier/models/tree_model.joblib
```
**These 6 hashes are the integrity check for the rest of this session — re-verified in R7. None of these files will be overwritten; any v2 artifact gets a new filename.**

## git status
Clean at session start.

## Python / package versions
```
Python 3.10.12
sklearn 1.7.2, numpy 1.26.4, pandas 2.3.3, scipy 1.8.0 (pre-existing NumPy-version warning on import, harmless), cv2 4.9.0, joblib 1.5.3, statsmodels 0.14.6
torch 2.13.0+cu130, CUDA available: True
```

## pytest baseline
```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --ignore=lerobot_mujoco_sim
no tests ran in 0.11s
```
Same as both prior sessions' baselines — zero pytest-discoverable tests
outside `lerobot_mujoco_sim` (which still can't collect: missing `mujoco`/
`pyarrow`, pre-existing, unrelated). Plain `pytest` still needs
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` due to an `anyio` plugin incompatibility
(pre-existing environment issue, not this session's).

## Capability re-checks (things the last two sessions found blocked)

**Gazebo headless**: available and working.
```
$ gz sim --version
Gazebo Sim, version 8.14.0
```

**DA-V2 checkpoints**: exist, machine-wide search, all OUTSIDE this repo:
```
/home/manasi/bps-project/checkpoints/depth_anything_v2_metric_hypersim_vits_NewModified.pth
/home/manasi/bps-project/checkpoints/depth_anything_v2_metric_hypersim_vits_NewModified1.pth
/home/manasi/bps-project/checkpoints/depth_anything_v2_vits.pth
```
`models/depth_anything_v2/dpt.py` (the non-metric/"relative" DA-V2
architecture) IS committed in this repo and importable
(`scripts/run_depth_anything.py`'s own usage pattern: `DepthAnythingV2(**cfg)`,
`load_state_dict(torch.load(checkpoint_path))`, `model.infer_image(image)`
— no ROS/rclpy needed for this path at all). The METRIC checkpoints exist
but their required import path, `metric_depth.depth_anything_v2.dpt`, does
**not** exist anywhere in this repo (`find . -iname metric_depth` empty) —
so only the non-metric/relative DA-V2 model (`depth_anything_v2_vits.pth`)
is actually usable here without additional missing code. Noted as a
**correction to two prior sessions' "DA-V2 is blocked" framing** — it's
blocked only via the ROS2 (`depth_publisher.py`) path, not via direct
Python inference (`run_depth_anything.py`'s pattern). `torch` is installed
with CUDA. This will be used in R2/R4 to generate a genuinely real (not
ray-cast) secondary depth-source feature set, closing a gap flagged twice
before.

**rclpy**: still not installed, re-confirmed (`ModuleNotFoundError`). Not
needed for the direct-inference path above.

**Important caveat for later use**: `depth_anything_v2_vits.pth` is the
non-metric/RELATIVE model (`infer_image(image)` single-argument signature,
matching `depth_publisher.py`'s documented `model_type=relative` path) —
its output is relative, unitless depth, NOT metric Z-depth in meters. Any
feature that assumes metric depth (`relative_height`, `minimum_boundary_distance`
when `is_metric=True`, `depth_to_world_points`'s scaling) will be on a
different, arbitrary scale when fed this output. This will be documented
plainly wherever DA-V2 features are used below, not silently treated as
equivalent to the ray-cast/GT metric depth.

## Reproduce this snapshot
```
git rev-parse HEAD
find . -iname "*.joblib" -o -iname "training_data*.csv" | sort | grep -v __pycache__ | xargs sha256sum
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --ignore=lerobot_mujoco_sim
gz sim --version
find / -iname "*.pth" 2>/dev/null
python3 -c "import rclpy"
```
