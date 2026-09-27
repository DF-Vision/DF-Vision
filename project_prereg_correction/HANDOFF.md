# Handoff — `project/prereg-correction`

Branch `project/prereg-correction`, from `project/audit` (`9b917d8`). Not
merged, not pushed. This resolves the open question
`project_audit/C6_AUDIT.md` raised about commit `0d74cd4`.

## Task 1 verdict, in three sentences

Commit `0d74cd4` bundled two changes that are demonstrably separable — a
necessary fix for a null-data crash bug, and a general decoupling of
`gap_signal_multiplier` from `noise_scale` that also happened to affect
PREREG.md's "2x effect" power scenario, which the commit message never
mentions. Reconstructing an isolated version that fixes only the crash
bug reproduces PREREG's stated 0.79-0.80 power almost exactly, but this
audit traced why: the old coupling made a stronger multiplier
simultaneously inflate BOTH the failure-logit coefficient AND
`Gap_initial`'s own measurement noise/spread, a compounding effect that
`power_analysis_c1.py`'s own docstring (dated ~7 weeks before the fix,
unmodified since) explicitly says should not happen — the two things are
supposed to vary independently. The current, post-fix behavior is
therefore the methodologically correct one, and PREREG's "2x → 0.80"
fallback should be treated as stale, not as an alternative valid number.

## Corrected power table (reduced scale, n_reps=100, current/correct generator)

| N | 1x | 2x | 3x |
|---|---|---|---|
| 90 | 0.040 | 0.150 | 0.330 |
| 120 | 0.100 | 0.160 | 0.400 |
| 180 | 0.060 | 0.310 | 0.600 |
| 195 | 0.100 | 0.340 | 0.600 |

No cell in this range reaches the 0.80 target. Full table (including the
old-coupling numbers for reference) in
`project_prereg_correction/PREREG_AMENDMENT_DRAFT.md`.

## Draft amendment's one-line recommendation

**Keep `0d74cd4`'s noise-decoupling change; do not revert it.** Correct
`PREREG.md`'s stated "2x → 0.79-0.80" figure to "2x → ~0.31-0.34 at
N=180-195" instead.

## What you need to do by hand

1. **Read `project_prereg_correction/PREREG_AMENDMENT_DRAFT.md` in full**
   and decide whether to accept its correction.
2. **If you accept it, edit `PREREG.md` yourself** — this session
   deliberately did not touch it. Replace the "approximately 0.79-0.80"
   sentence in the C1 power-justification paragraph with the corrected
   number, and revise the "Pre-registered stance on this uncertainty"
   paragraph's "meeting the target" framing, since no tested scenario in
   the 1x-3x/N=90-195 range currently meets it.
3. **Separately decide whether `0d74cd4`'s noise-model change itself
   needs any further adjustment** — this audit recommends keeping it
   as-is (it matches this project's own older documented intent), but
   that is a recommendation, not a code change this session made. If you
   disagree with the reasoning in `COMMIT_DIAGNOSIS.md` §6-7, that would
   be the place to push back.
4. **The bigger, unresolved fact stands regardless of which noise model
   is "correct"**: zero real trials exist, `b_gap=2.5` is an arbitrary
   placeholder, and this whole power analysis is provisional until real
   pilot data lets you replace it with a measured effect size.

## Task 4 — Integrity check

**SHA256 of protected files vs. C0 baseline** (`project_closure/LOG.md`):
```
$ find . -iname "*.joblib" | sort | xargs sha256sum
$ sha256sum integration/prg_config.json perception-calibration-gap/PREREG.md "SCHEMA.md " "INTERFACE.md "
```
All 13 hashes byte-identical to the C0 baseline (checked line-by-line).
`integration/prg_config.json`: `{ "use_stub": true }`.
`git diff project/overnight-hardening..HEAD -- integration/prg_config.json`:
empty output — confirmed via git diff.

**`git diff project/audit..HEAD --stat`:**
```
$ git diff project/audit..HEAD --stat
 project_prereg_correction/COMMIT_DIAGNOSIS.md          | ...
 project_prereg_correction/HANDOFF.md                   | ...
 project_prereg_correction/PREREG_AMENDMENT_DRAFT.md    | ...
 project_prereg_correction/scripts/full_power_table.py  | ...
 project_prereg_correction/scripts/run_isolated_check.py| ...
 5 files changed, ... insertions(+)
```
Only `project_prereg_correction/` files — no other file in the repo
touched. `PREREG.md`/`SCHEMA.md `/`INTERFACE.md ` confirmed untouched
(separately, via `git diff --stat` on those three paths: empty).

**Full test suite:**
```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q
146 passed, 1 skipped
```
Matches the last known count exactly (146 passed, 1 skipped, from the
closure/audit sessions).
