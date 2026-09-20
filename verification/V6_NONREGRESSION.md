# V6 — Non-regression and Reproducibility

Since V2 = STOP, no `boundary_mode` switch was implemented (V3 skipped),
so V4 (independent-oracle re-validation of a fix) and V5 (end-to-end chain
in pairwise mode) have nothing to validate and are also skipped, per
`V2_GATE.md`. V6 is adapted accordingly: it confirms this session made no
regressions and that what WAS produced (the V1 audit scripts and data) is
reproducible — not that a nonexistent fix is regression-free.

## V6a. pytest

```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --ignore=lerobot_mujoco_sim
no tests ran in 0.11s
```
Identical to the V0 baseline (`verification/BASELINE.md`). No new test
file was added (there is no fix for `tests/test_boundary_mode.py` to test),
so there is nothing to compare against a "new failures" bar — the
repo's pytest state is byte-for-byte the same as before this session.

## V6b. Model/scaler/config/PREREG integrity

```
$ find . -iname "*.joblib" | sort | xargs sha256sum
b38860a8797fc78b7c59bbb9980d49f5b4b67545a661f8bf17ae091248ea16c1  ./relation_classifier/models/logistic_model.joblib
f0d864a06fa6154a6d68a71a36b00a4bd1aa03cdf5e2ebd712cd0eb3cca760c3  ./relation_classifier/models/scaler.joblib
725e95e4eafdaadab88abe35adf0b77f97f16ba0c572c87f885bc1591ae91793  ./relation_classifier/models/support_logistic_model.joblib
66d65fe827a4469cb70625b49a38be8338ba13fccb0ebef7a10008741b2efd10  ./relation_classifier/models/support_scaler.joblib
12dfc1f85ad49084318d5f7bd7dca7fb72f7c9b8c837e91da1996b8255e17a6c  ./relation_classifier/models/tree_model.joblib
```
**Identical to V0 baseline, all 5 files, confirmed via `diff`.**

```
$ git diff unified-pipeline..HEAD --stat -- '*.joblib' integration/prg_config.json perception-calibration-gap/PREREG.md perception-calibration-gap/mock-pipeline/conformal_gate.py perception-calibration-gap/mock-pipeline/gap_computer.py
(empty)
$ git diff overnight/prg-diagnosis..HEAD --stat -- '*.joblib' integration/prg_config.json perception-calibration-gap/PREREG.md
(empty)
```
No changes to any `.joblib`, `integration/prg_config.json`, `PREREG.md`,
or Person-A's `conformal_gate.py`/`gap_computer.py`, against either the
`unified-pipeline` base or this session's actual branch point.

## V6c. Fresh-clone reproducibility

```
$ git clone --quiet /home/manasi/DF-Vision <tmpdir>
$ cd <tmpdir> && git checkout fix/boundary-features --quiet
```
Ran, in the fresh clone, with no environment setup beyond what was
already on this machine (no README install step exists for this
subproject beyond the repo-root `requirements.txt`, already installed):

- `verification/scripts/v1f_rerun_with_real_gazebo_contact.py` — reproduced
  **byte-for-byte identical** output (balanced accuracy 0.5000, same CI,
  same per-class recall, same n) using the committed
  `verification/gazebo_contacts_SIM/*.csv` files, no gz sim needed.
- `verification/scripts/quantify_ring_mismatch.py` — reproduced
  **byte-for-byte identical** per-pair ring pixel counts and
  boundary_sharpness values.
- `verification/scripts/run_gazebo_contacts_SIM.py world1_baseline.sdf` —
  re-ran the actual headless-gz-sim-dependent script from the fresh clone
  and got the same result (contact=1: 2, unknown: 0) as the original run.

No hidden local-state problems found (no absolute paths outside the repo,
no uncommitted-but-required files) — every script uses
`Path(__file__).resolve().parents[N]`-relative paths throughout. Fresh
clone removed after the check.

## V6d. daily_integration_check.py

```
$ cd perception-calibration-gap && python3 daily_integration_check.py
...
=== Daily Integration Check: PASS ===
```
Still green, unaffected by this session (this session touched only
`relation_classifier/` diagnostics and `verification/`/new files, nothing
`daily_integration_check.py` depends on). `git status --short` confirmed
clean after the run (the script's own internal log-file reset touches
only its own gitignored/untracked scratch JSONL, not a tracked file).

## V6e. Determinism

The V1f rehearsal ran twice with identical results: once in the main
working tree, once in the fresh clone (V6c above) — same balanced
accuracy, same CI, same per-class recall, same 43-row table. This
satisfies the determinism check by construction (two independent runs,
two different working directories, identical output), standing in for a
literal "run V5 twice" since V5 doesn't apply (no fix implemented).

## Conclusion

This session made zero changes to model/scaler/config/PREREG files,
introduced no pytest regressions (there were none to introduce — no fix
code was written), left `daily_integration_check.py` green, and every
diagnostic script this session added is reproducible from a fresh clone
with no hidden state.
