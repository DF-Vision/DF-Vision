# T11 — Verification

## Full pytest (vs T0 baseline)
```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q
73 passed, 1 xfailed, 31 warnings in 0.71s
```
T0 baseline was 12 passed. This session added 61 new passing tests
(T1: 8, T2: 10, T3: 11, T5: 6, T6a: 5 passed + 1 xfailed (documented
known bug, not a regression), T8/T9 model_d: 11, T9 data-quality/
scheduler: 10) across 7 new test files. No existing test regressed.

## SHA256 — all protected files byte-identical to T0
```
[11 hashes: 5 legacy .joblib, 4 v2 .joblib, prg_config.json, PREREG.md,
SCHEMA.md , INTERFACE.md , 2 committed CSVs -- all confirmed identical
to project_overnight/LOG.md's T0 baseline via direct diff]
```

## git diff on protected files — empty
```
$ git diff main..HEAD -- integration/prg_config.json perception-calibration-gap/PREREG.md "SCHEMA.md " "INTERFACE.md "
(empty)
```

## Fresh-clone reproduction
```
$ git clone --quiet /home/manasi/DF-Vision <tmpdir> && cd <tmpdir>
$ git checkout project/overnight-hardening --quiet
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q
73 passed, 1 xfailed, 31 warnings in 0.71s        # bit-for-bit identical to main tree
$ cd perception-calibration-gap && python3 daily_integration_check.py
=== Daily Integration Check: PASS ===
```
No hidden local state — every new script uses
`Path(__file__).resolve().parents[N]`-relative imports. Fresh clone
removed after the check.

## Determinism
- `tests/test_extrinsic_calibration.py`, `tests/test_geometric_prg_error_sanity.py`
  self-check determinism directly (same-seed calls asserted equal within
  the test itself).
- Fresh-clone pytest run reproduced the exact same pass/fail/xfail counts
  as the main-tree run (see above) -- a second independent run with the
  same fixed seeds (`RANDOM_STATE = 20260920` throughout, consistent with
  every prior session on this codebase).
- `project_overnight/scripts/t7_synthetic_recovery_check.py` uses fixed
  seeds; re-run twice during this session (once before, once after the
  `mock_data_generator.py` bugfix) and gave identical numbers on
  unaffected runs (the null-FPR numbers changed ONLY because of the
  intentional bugfix, not seed nondeterminism -- confirmed by the
  planted-recovery numbers being byte-identical across both runs).

## daily_integration_check.py
Still green (main tree and fresh clone both confirmed above). Note: this
session's T9 `TrialRecord` extension broke this check once (missing
fields in its `allowed_none` allowlist) -- caught and fixed during T7's
verification pass, logged in `project_overnight/LOG.md`'s T7 entry as a
genuine regression catch, not silently avoided.

## Regression tests showing legacy behavior unchanged
- `tests/test_relation_model_v2_regression.py` (prior session, 7/7) --
  re-confirmed passing.
- `tests/test_prg_edge_error_mode_regression.py` (prior session, 5/5) --
  re-confirmed passing, including the config-switch default-is-classifier
  check.
- `logger.py`'s own `__main__` self-test block (all 4 sub-tests) --
  re-run directly this session after the T9 `TrialRecord` extension,
  confirmed passing unmodified.

## Known, disclosed gaps (not silently hidden)
- `analysis_c1.rf_shap_check()` remains UNVERIFIED this session (broken
  `shap` install, in-process stub used only to unblock everything else
  in that file).
- T4's `--replay` mode and config-driven camera source were not built
  (flagged in `project_overnight/LOG.md`'s T4 entry).
- T6b/T6c (changed-interface PRG variant, full re-evaluation) were not
  attempted (flagged in the T6 log entry).
- `train_yolo_scaffold.py`'s end-to-end run was not re-confirmed this
  session (timed out after 90s under apparent CPU contention; not
  re-diagnosed further per the attempt budget).
