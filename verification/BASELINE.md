# V0 — Baseline Snapshot

Branch `fix/boundary-features` created from `overnight/prg-diagnosis` @ `0a51fbb`.
This file records the state BEFORE any change in this session, so later
results are attributable.

## HEAD commit

```
$ git rev-parse HEAD
0a51fbb1d46de08ce959674826dfc61e60407519
```
(This is the commit `fix/boundary-features` was branched from; no commits
made yet at time of this snapshot.)

## git status

```
$ git status --short
(clean)
```

## SHA256 of every .joblib/scaler file

```
$ find . -iname "*.joblib" | sort | xargs sha256sum
b38860a8797fc78b7c59bbb9980d49f5b4b67545a661f8bf17ae091248ea16c1  ./relation_classifier/models/logistic_model.joblib
f0d864a06fa6154a6d68a71a36b00a4bd1aa03cdf5e2ebd712cd0eb3cca760c3  ./relation_classifier/models/scaler.joblib
725e95e4eafdaadab88abe35adf0b77f97f16ba0c572c87f885bc1591ae91793  ./relation_classifier/models/support_logistic_model.joblib
66d65fe827a4469cb70625b49a38be8338ba13fccb0ebef7a10008741b2efd10  ./relation_classifier/models/support_scaler.joblib
12dfc1f85ad49084318d5f7bd7dca7fb72f7c9b8c837e91da1996b8255e17a6c  ./relation_classifier/models/tree_model.joblib
```
**These 5 hashes are the integrity check for the rest of this session.
Re-verified in V6; any change is a hard failure of rule 3.**

## Python / package versions

```
$ python3 --version
Python 3.10.12

sklearn 1.7.2
numpy 1.26.4
pandas 2.3.3
scipy 1.8.0
cv2 4.9.0
joblib 1.5.3
statsmodels 0.14.6
```
Note: `scipy` prints a compatibility warning on import ("A NumPy version
>=1.17.3 and <1.25.0 is required for this version of SciPy (detected
version 1.26.4)") on every invocation in this environment. Pre-existing,
not caused by this session, harmless in every run so far (no scipy
functionality used here has broken), but noting it since "SciPy version
mismatch" could otherwise look like a new problem introduced by this work.

## pytest baseline — IMPORTANT, read before trusting any "pytest passes" claim later

`pytest` (v6.2.5) is broken as invoked plainly in this environment: it
fails at plugin-load time with `ModuleNotFoundError: No module named
'_pytest.scope'`, triggered by the installed `anyio` (v4.14.2) package's
pytest plugin, which expects a newer pytest internal API. This is a
pre-existing environment issue, not caused by this session. Workaround
used for the rest of this session: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

```
$ python3 -m pytest --collect-only -q
ModuleNotFoundError: No module named '_pytest.scope'  (via anyio's pytest plugin)

$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest --collect-only -q
ERROR collecting lerobot_mujoco_sim/run_mujoco_grasp_smoke_test.py
  ModuleNotFoundError: No module named 'mujoco'
ERROR collecting lerobot_mujoco_sim/scripts/tests/test_parquet.py
  ModuleNotFoundError: No module named 'pyarrow'
Interrupted: 2 errors during collection
```
Both are pre-existing, unrelated to relation_classifier/PRG work (missing
third-party packages for an entirely different subsystem — MuJoCo sim
data collection). Not fixed here, out of scope for this task (rule 5/6
don't cover unrelated subsystems, and fixing an unrelated env-dependency
gap isn't part of the boundary-features fix).

```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --ignore=lerobot_mujoco_sim
no tests ran in 0.11s
```
**Baseline fact, stated plainly: this repository currently has ZERO
pytest-discoverable tests outside `lerobot_mujoco_sim` (which itself
can't collect, see above).** `perception-calibration-gap/test_real_adapter.py`
and `perception-calibration-gap/mock-pipeline/test_prg_stub_integration.py`
are script-style files (`def main(): ... if __name__ == "__main__":`),
not pytest test modules — confirmed via `grep -n "^def \|^if __name__"`,
no `def test_*` in either file. So "pytest passes" is not yet a meaningful
regression signal for this repo; `tests/test_boundary_mode.py` (V3) will
be the first real pytest file touching this code path. V6a's "pytest
still passes" claim will mean exactly and only: `tests/test_boundary_mode.py`
passes, plus zero new collection errors introduced elsewhere by this
session's changes — not a general regression guarantee across the whole
repo, because the repo doesn't have one.

## Reproduce this snapshot

```
git rev-parse HEAD
find . -iname "*.joblib" | sort | xargs sha256sum
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --ignore=lerobot_mujoco_sim
```
