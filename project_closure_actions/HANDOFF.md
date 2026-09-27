# Handoff — `project/closure-actions`

Branch `project/closure-actions`, created from `project/final-integration`.
Not merged into anything, not pushed. A separate branch,
`fix/readme-status-correction` (from `origin/failure-calibration`), was
also created this session for G4 — also not merged, not pushed; see its
own section below.

## G1-G4 confirmed done, exact diffs quoted

### G1 — PREREG power-analysis amendment applied

```
$ git diff project/final-integration..HEAD --stat -- perception-calibration-gap/PREREG.md
 perception-calibration-gap/PREREG.md | 130 ++++++++++++++-
```
Pure append, verified byte-for-byte: the pre-G1 file content is an exact
prefix of the post-G1 file (`diff` of the two, empty). New section:
`## Amendment 2026-09-27 — Power analysis correction`, using
`project_prereg_correction/PREREG_AMENDMENT_DRAFT.md`'s reviewed content,
plus the PRG-retirement note as its final paragraph. The original
"0.79-0.80" paragraph is untouched, per `PREREG.md`'s own Section 7 rule.

### G2 — status documents updated

```
$ git diff project/final-integration..HEAD --stat -- project_final/PROJECT_STATUS_FINAL.md project_final/PRG_SIGNAL_STATUS.md project_final/RESOLUTION_LOG.md
 project_final/PRG_SIGNAL_STATUS.md    | 18 ++--
 project_final/PROJECT_STATUS_FINAL.md | 15 ++
 project_final/RESOLUTION_LOG.md       | 28 ++--
```
`PROJECT_STATUS_FINAL.md`: new "Addendum — 2026-09-27" section below the
table (table itself untouched). `PRG_SIGNAL_STATUS.md`: closing paragraph
now states the retirement decision instead of leaving it open.
`RESOLUTION_LOG.md`: rows 6 and 11 marked RESOLVED with pointers; summary
counts corrected (was "2 OPEN, 5 not-applied, 6 resolved" → now "1 OPEN,
4 not-applied, 8 resolved").

### G3 — `e5_analyze_real_capture_gaps.py` made pluggable

```
$ git diff project/final-integration..HEAD --stat -- explore/scripts/e5_analyze_real_capture_gaps.py tests/test_e5_candidate_harness.py
 explore/scripts/e5_analyze_real_capture_gaps.py | 201 +++++++++++++++++-------
 tests/test_e5_candidate_harness.py              | 144 +++++++++++++++++
```
New `--candidate {c1,c2,c3,c5-neighbor-aware,all}` flag (default `all`),
registry-based dispatch. c1/c2/c3's legacy computation and output values
verified byte-identical to git HEAD's pre-change script on an identical
synthetic fixture (8/8 new tests, including a hard-coded golden-output
regression check). c5-neighbor-aware gets its own native
estimated-gap-based error columns, does not touch legacy columns. **No
real capture data was run; no performance number is claimed anywhere for
c5-neighbor-aware** — only that the harness can now run it.

### G4 — README correction prepared, NOT pushed

Located: commit `c19ecc7` (`origin/failure-calibration`, author
`krupa-shah23`), the file `README.md`. Created local branch
`fix/readme-status-correction` from `origin/failure-calibration` (fetched
first), corrected the "Current Status" section's false claims (was:
"✅ PRG Integration," "✅ Hardware Pipeline," "✅ Voice Interaction" —
none ever true), and pointed to `project_final/PROJECT_STATUS_FINAL.md`
on `project/final-integration` as the authoritative source instead of
restating status inline. Committed locally only:
```
$ git log --oneline fix/readme-status-correction -2
4de0c3d Correct README's stale/false Current Status section
c19ecc7 Added ReadME
```
Diff is scoped exactly to the "Current Status" section (33 insertions, 7
deletions, nothing else in the file touched).

**The exact `git push` command that would push it** (verified safe via
`git push --dry-run`, which itself pushes nothing — the dry-run confirmed
this branch's upstream tracking is `origin/failure-calibration`, meaning
a BARE `git push` or `git push origin HEAD:failure-calibration` would
push directly onto the shared branch, which is NOT what should happen):

```bash
git checkout fix/readme-status-correction
git push origin HEAD
```

This creates a NEW remote branch `fix/readme-status-correction` (does
NOT touch `origin/failure-calibration` itself). **Reminder: `failure-calibration`
is a branch Krupa also uses — review the diff above before pushing
anything, and push to the new branch name shown, not to `failure-calibration`
directly.**

## No PRG design work, no retraining, no new experiments happened this session

This session executed two already-made decisions (documented them,
did not make them) and one small, well-scoped code change (a CLI
dispatch mechanism, not a new algorithm or evaluation). Specifically:
- No `.joblib`/scaler/committed dataset was touched (verified: SHA256 of
  all 5 `.joblib` files and `integration/prg_config.json` identical to
  `project_final/BASELINE.md`'s F0 values — only `PREREG.md`'s hash
  differs, the one authorized append).
- `use_stub` in `integration/prg_config.json` was never read, tested, or
  argued about this session — confirmed via `git diff`, empty.
- G3's harness change was verified to run without error on synthetic
  data only; zero performance numbers were generated or reported for
  c5-neighbor-aware anywhere in this repo.
- The two decisions in this task's preamble (PRG retirement, Track C's
  future framing) were stated as already-made facts and only recorded,
  not re-derived or re-argued.

## G5 — Verification (full detail)

```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q
154 passed, 1 skipped, 305 warnings in 57.64s
```
(146 from `project_final/BASELINE.md`'s F0 baseline + 8 new G3 tests = 154
— exact, accounted for.) The 1 skip is the same pre-existing,
environment-conditional `shap` skip as every prior session.

SHA256: all 5 `.joblib` files and `integration/prg_config.json` identical
to F0. `PREREG.md`'s hash differs (expected — the one authorized G1
append); `SCHEMA.md `/`INTERFACE.md ` identical (neither touched this
session).

```
$ git diff project/final-integration..HEAD --stat
 explore/scripts/e5_analyze_real_capture_gaps.py | 201 +++++++++++++++++-------
 perception-calibration-gap/PREREG.md            | 130 ++++++++++++++-
 project_final/PRG_SIGNAL_STATUS.md              |  18 ++-
 project_final/PROJECT_STATUS_FINAL.md           |  15 ++
 project_final/RESOLUTION_LOG.md                 |  28 ++--
 tests/test_e5_candidate_harness.py              | 144 +++++++++++++++++
 6 files changed, 464 insertions(+), 72 deletions(-)
```
Exactly the intended files — no protected file, no unrelated file. (This
diff correctly does not show `fix/readme-status-correction`'s changes,
since that's a separate branch built on unrelated history, not a
descendant of `project/final-integration`.)

`use_stub`: confirmed untouched (`git diff` on `integration/prg_config.json`,
empty). `PREREG.md`: confirmed pure append (byte-identical prefix check).

## The honest bottom line

The only work left on this project that isn't already done is physical
— see `project_final/MASTER_HARDWARE_CHECKLIST.md`. This session closed
out the last remaining software loose ends (applying an already-reviewed
amendment, recording an already-made decision, and a small harness
readiness improvement) without touching the science. **No further
autonomous session should touch this repo's software until real capture
or trial data exists** — this was already true at the end of
`project/final-integration` and remains true now; nothing this session
did changes that.
