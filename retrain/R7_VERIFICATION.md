# R7 — Verification

## Full pytest (vs R0 baseline)

```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --ignore=lerobot_mujoco_sim
.......                                                                  [100%]
7 passed, 1 warning in 0.52s
```
R0 baseline was "no tests ran" (zero pytest-discoverable tests existed).
This session added `tests/test_relation_model_v2_regression.py` (7 tests,
all passing) — the repo's first real pytest suite. No new failures
introduced anywhere else; `lerobot_mujoco_sim`'s pre-existing collection
errors (missing `mujoco`/`pyarrow`) are unchanged and unrelated.

## SHA256 of legacy artifacts — identical to R0

```
$ find . -iname "*.joblib" -o -iname "training_data*.csv" | sort | grep -v __pycache__ | xargs sha256sum
079901037563e0887b8fbdba9fdf74ecf13b9da64936ddd1b18bc020b6ba0a26  ./relation_classifier/data/training_data_metric_only.csv
b38860a8797fc78b7c59bbb9980d49f5b4b67545a661f8bf17ae091248ea16c1  ./relation_classifier/models/logistic_model.joblib
f0d864a06fa6154a6d68a71a36b00a4bd1aa03cdf5e2ebd712cd0eb3cca760c3  ./relation_classifier/models/scaler.joblib
725e95e4eafdaadab88abe35adf0b77f97f16ba0c572c87f885bc1591ae91793  ./relation_classifier/models/support_logistic_model.joblib
66d65fe827a4469cb70625b49a38be8338ba13fccb0ebef7a10008741b2efd10  ./relation_classifier/models/support_scaler.joblib
12dfc1f85ad49084318d5f7bd7dca7fb72f7c9b8c837e91da1996b8255e17a6c  ./relation_classifier/models/tree_model.joblib
```
Identical, byte-for-byte, to `retrain/BASELINE.md`'s recorded hashes.
New v2 artifacts (`retrain/models/relation_binary_v2*.joblib`,
`support_v2*.joblib`) are ADDITIONS under new filenames, never overwrites
— confirmed via `git diff` showing only new files, never a hash change on
an existing path.

## git diff on protected files — empty

```
$ git diff HEAD -- integration/prg_config.json perception-calibration-gap/PREREG.md perception-calibration-gap/mock-pipeline/conformal_gate.py perception-calibration-gap/mock-pipeline/gap_computer.py
(empty)
$ cat integration/prg_config.json
{ "use_stub": true }
```
`relation_classifier/relation_model_config.json` (the NEW, separate
config file this session created for the version switch) is a different
file entirely — `integration/prg_config.json` was never touched.

## Fresh-clone reproduction

```
$ git clone --quiet /home/manasi/DF-Vision <tmpdir>
$ cd <tmpdir> && git checkout retrain/relation-classifier --quiet
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --ignore=lerobot_mujoco_sim
7 passed, 1 warning in 0.52s
$ python3 retrain/scripts/r3_train_and_select.py
[... identical CV numbers to the original run, see determinism below ...]
```
No hidden local-state problems found — every script uses
`Path(__file__).resolve().parents[N]`-relative paths, and
`classify_relation.py`'s CWD-relative legacy model loads work identically
in the fresh clone (invoked from its own root, matching the existing
convention). Fresh clone removed after the check.

## Determinism — two seeded runs identical

R3's fresh-clone re-run produced **bit-for-bit identical** CV numbers to
the original run in the main working tree (same means, same std, same
per-fold scores, same selected candidate `two_stage_logreg`) — confirmed
by direct comparison of both runs' printed output. `RANDOM_STATE=20260920`
is fixed and used consistently across data generation (R2), training
(R3), and the robustness sweep (R4).

## daily_integration_check.py

```
$ cd perception-calibration-gap && python3 daily_integration_check.py
=== Daily Integration Check: PASS ===
```
Still green, unaffected (this session never touched anything
`daily_integration_check.py` depends on).

## Hidden local state — none found

Checked for: absolute paths outside the repo (none — the one genuine
external dependency, the DA-V2 checkpoint at `/home/manasi/bps-project/checkpoints/...`,
is explicitly logged as such in `retrain/BASELINE.md` and
`retrain/scripts/r2b_generate_da_v2_features.py`'s own `CHECKPOINT`
constant, not silently assumed); uncommitted-but-required files (none —
`retrain/data/relation_dataset_v2_SIM.csv` and its manifest are committed);
environment-variable dependencies beyond the documented
`OMP_NUM_THREADS`-family thread-limiting (used only to reduce CPU
contention on this session's heavily-loaded machine, not required for
correctness — every script runs fine without them, just slower).
