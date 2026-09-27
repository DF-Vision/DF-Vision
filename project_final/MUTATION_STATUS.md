# F5 — Mutation-Testing Status

## What F1-F4 actually added

```
$ git diff project/prereg-correction..HEAD --stat
 project_final/BASELINE.md           |  65 +++++++++++++++++++
 project_final/POWER_STATUS.md       |  84 ++++++++++++++++++++++++
 project_final/PRG_SIGNAL_STATUS.md  | 126 ++++++++++++++++++++++++++++++++++++
 project_final/RESOLUTION_LOG.md     |  40 ++++++++++++
 project_final/TEST_CONSOLIDATION.md | 121 ++++++++++++++++++++++++++++++++++
 5 files changed, 436 insertions(+)
```
**Zero code was added or cherry-picked in F1-F4** — only consolidation
documentation. The branch-ancestry investigation (F-branch step) found
exactly 2 non-ancestor commits and rejected both (one superseded, one
conflicting with verified findings — see `RESOLUTION_LOG.md` row 1);
neither was cherry-picked. `TEST_CONSOLIDATION.md` (F4) root-caused a
pre-existing bug in a standalone test script but did not fix it (out of
scope, logged as such). **There is therefore nothing new this session
introduced that requires its own mutation test.**

## Consolidated re-verification of the existing 30/30 kill rate

Re-ran the unmodified mutation harness from `project/closure` (C7) on
this integrated branch, to confirm the 30/30 result still holds after
consolidation (not a new experiment — a re-verification of an existing,
already-established result):
```
$ python3 project_closure/scripts/c7_run_mutations.py
...
Kill rate: 30/30
```
**No regression.** All 30 mutations across the 10 named guards
(calibration solver, lookup-table/out-of-reach, dry-run/
`--i-am-supervising`, marker-size config, template-CSV guard, Model D
stub-exclusion, session scheduler, `data_quality_report`,
controller-freeze check, depth-unit guard) are still killed. Working
tree confirmed clean (`git status --porcelain`, empty) after the
harness's own mutate/revert cycle — no mutation was left applied.
