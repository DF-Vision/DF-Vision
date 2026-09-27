# F8 — Verification

## Full test suite

```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q
146 passed, 1 skipped, 304 warnings in 55.76s
```
Identical to F0's baseline count.

## SHA256 of protected files vs. F0 baseline

All 13 hashes (5 `.joblib` files, `integration/prg_config.json`,
`perception-calibration-gap/PREREG.md`, `SCHEMA.md `, `INTERFACE.md `)
re-computed and diffed programmatically against `project_final/BASELINE.md`'s
recorded values — **identical, zero differences**.

## `git log`/`git diff` integrity

```
$ git diff project/prereg-correction..HEAD --stat -- perception-calibration-gap/PREREG.md "SCHEMA.md " "INTERFACE.md "
(empty)
```
No direct edit to any of the 3 frozen documents anywhere in this
session's history — confirmed by diffing the FULL range of this
session's commits against the branch point, not just the final state
(rules out an edit-then-revert within the session that a final-state-only
diff would hide).

```
$ git diff project/prereg-correction..HEAD --stat
 PROJECT_STATUS.md                          |  15 +++
 project_final/BASELINE.md                  |  65 ++
 project_final/MASTER_HARDWARE_CHECKLIST.md | 164 ++
 project_final/MUTATION_STATUS.md           |  40 ++
 project_final/POWER_STATUS.md              |  84 ++
 project_final/PRG_SIGNAL_STATUS.md         | 126 ++
 project_final/PROJECT_STATUS_FINAL.md      |  45 ++
 project_final/RESOLUTION_LOG.md            |  40 ++
 project_final/TEST_CONSOLIDATION.md        | 121 ++
 9 files changed, 700 insertions(+)
```
Exactly 1 append-only edit (`PROJECT_STATUS.md`, the F7-logged pointer
note, appended not rewritten) plus 8 new `project_final/` files. Nothing
else in the repo touched, no line deleted anywhere.

`git log --oneline project/prereg-correction..HEAD`: 8 sequential,
forward-only commits (F0-F7), each on top of the last. This is a local
branch, never pushed to any remote (`git branch -a` shows no
`remotes/origin/project/final-integration`) — no force-push is possible
to have occurred. No `git reset`, `rebase`, or `filter-branch` was run
this session (only `git add`/`commit`/`checkout -b`/read-only `git
show`/`diff`/`merge-base` commands were used throughout — reviewable in
this session's own command history).

## Fresh-clone reproduction

```
$ git clone <this repo> <tmpdir> && cd <tmpdir> && git checkout project/final-integration
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q
146 passed, 1 skipped   -- IDENTICAL
$ python3 perception-calibration-gap/daily_integration_check.py            # PASS
$ python3 perception-calibration-gap/e2e_integration_rehearsal_SYNTHETIC.py  # PASS, all 7 stages
$ python3 perception-calibration-gap/real_camera/capture_and_gap.py --replay .../fixtures/replay_synthetic_v1  # 4/4 good + 4/4 correctly-flagged
$ python3 -c "... c6.variance_equality_check(seed=1) ..."                  # (True, 0.0026..., 0.0026...) -- identical
```
All green from a clean clone at this exact commit. Temp clone removed
after verification (`rm -rf`), no artifacts left outside the repo.

## Working tree

`git status --porcelain`: clean before and after this entire F8 pass.
