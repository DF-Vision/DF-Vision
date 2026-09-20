# E4 — Non-regression and Reproducibility

## Full pytest (vs E0 baseline)
```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --ignore=lerobot_mujoco_sim
12 passed, 2 warnings in 0.52s
```
E0 baseline was 7 passing (from the prior session's
`test_relation_model_v2_regression.py`). This session added
`tests/test_prg_edge_error_mode_regression.py` (5 tests) -> 12 total,
all passing. No regressions; this is expected growth, not a discrepancy.

## SHA256 of all model/data artifacts — identical to E0
```
$ find . -iname "*.joblib" -o -iname "training_data*.csv" -o -iname "relation_dataset_v2_SIM.csv" | sort | grep -v __pycache__ | xargs sha256sum
[all 11 hashes byte-identical to explore/BASELINE.md -- confirmed via diff]
```

## git diff on protected files — empty
```
$ git diff HEAD -- integration/prg_config.json perception-calibration-gap/PREREG.md perception-calibration-gap/mock-pipeline/conformal_gate.py perception-calibration-gap/mock-pipeline/gap_computer.py
(empty)
$ cat integration/prg_config.json
{ "use_stub": true }
```
This session's two new config files
(`relation_classifier/relation_model_config.json` — from the prior
session, untouched here — and `relation_classifier/prg_edge_error_config.json`,
new this session) are separate from `integration/prg_config.json`, which
remains exactly `{ "use_stub": true }`.

## Fresh-clone reproduction
```
$ git clone --quiet /home/manasi/DF-Vision <tmpdir> && cd <tmpdir>
$ git checkout explore/geometric-prg-error --quiet
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --ignore=lerobot_mujoco_sim
12 passed, 2 warnings in 0.54s
$ python3 explore/scripts/e1_oracle_ceiling.py    # bit-for-bit identical to the main-tree run
$ python3 explore/scripts/check_structural_independence.py
RESULT: PASS
```
No hidden local-state problems — every script uses
`Path(__file__).resolve().parents[N]`-relative imports. Fresh clone
removed after the check.

## Two seeded runs — identical outputs
`explore/scripts/e1_oracle_ceiling.py` run twice (once in the fresh
clone, once again in the main tree) produced **bit-for-bit identical**
balanced accuracy, per-class recall, and confusion matrix numbers —
confirmed by direct diff. `explore/scripts/e3_evaluate_candidates.py`'s
determinism is additionally self-checked inside the script itself
(criterion (a) asserts `s1 == s2` for every candidate on every row,
computed twice in the same process — this assertion passed for all 263
rows x 3 candidates, or the run would have raised `AssertionError` and
exited non-zero, which it did not).

## daily_integration_check.py
```
$ cd perception-calibration-gap && python3 daily_integration_check.py
=== Daily Integration Check: PASS ===
```
Still green, unaffected by this session.

## Conclusion
No regressions. All new artifacts are additive (new files, new config
keys with safe defaults) — nothing existing was overwritten, deleted, or
behaviorally changed in its default configuration.
