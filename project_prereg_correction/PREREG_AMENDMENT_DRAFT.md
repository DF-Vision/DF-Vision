# PREREG.md — Amendment Draft — DRAFT, NOT YET APPLIED

Drafted 2026-09-27 (autonomous audit follow-up session), branch
`project/prereg-correction`. Per this project's own rule, `PREREG.md` is
never edited in place — this is drafted text for a human to review and
paste in by hand (or reject). `PREREG.md` itself has NOT been touched
(`git diff` on it is empty — confirmed in `project_prereg_correction/HANDOFF.md`'s
Task 4 check).

---

## What `PREREG.md` currently states (quoted verbatim, lines 50-71)

> **C1 (Gap_initial coefficient, Model B):** At the planned N (195) and
> the conservative/realistic detector-noise assumption already used
> elsewhere in this project (mock_data_generator.py's default noise
> scales), estimated power is approximately 0.12-0.15 -- well below the
> conventional 0.80 target. Achieving 0.80 power at this effect size
> would require approximately 3,000+ trials, which is not feasible given
> hardware and time constraints (Section 8, Cut Strategy). However, if
> the true Gap effect is at least ~2x the conservative default (a real
> possibility, not a guarantee, given the conservative default was
> itself a guess), estimated power at N=195 is approximately 0.79-0.80,
> meeting the target.

## What was found

`mock_data_generator.py` (the file this power estimate is explicitly
based on) was modified exactly once since `PREREG.md` was written —
commit `0d74cd4` (2026-09-21, this project's `T7` null-data bugfix). That
commit changed one line:
```diff
-noise_scale = OBJECT_DETECTOR_NOISE[obj] * gap_signal_multiplier
+noise_scale = OBJECT_DETECTOR_NOISE[obj]
```
This was a genuine, necessary bugfix: at `gap_signal_multiplier=0.0`
(the "null" scenario), the OLD line zeroed BOTH the effect coefficient
AND the measurement noise, producing a degenerate constant `Gap_initial`
column. The fix was correct for that case.

**However**, the same line change also altered every `gap_signal_multiplier
!= 1.0` scenario, including the `2x` scenario `PREREG.md`'s power
justification above depends on. An audit (`project_audit/C6_AUDIT.md`,
then this session's `project_prereg_correction/COMMIT_DIAGNOSIS.md`)
found:

1. **Reconstructing the pre-`0d74cd4` generator from git history and
   re-running the power simulation at N=195, 2x reproduces `PREREG.md`'s
   stated 0.79-0.80 exactly** (measured: 0.825 in the audit's first
   check; 0.800-0.810 in this session's reduced-scale, differently-seeded
   re-runs — consistent, not identical, as expected from Monte Carlo
   noise across different seeds/rep-counts).
2. **The two changes bundled in `0d74cd4` are separable**: a version
   that fixes ONLY the null-case bug (special-casing `mult=0.0`) while
   leaving every other multiplier's behavior exactly as it was before
   the fix reproduces the same 0.79-0.80-ish number
   (`project_prereg_correction/scripts/run_isolated_check.py`:
   N=195, mult=2.0 → power=0.800).
3. **But that reproduction is not evidence the pre-fix number was
   correct** — it comes from a mechanism this audit traced precisely:
   the pre-fix code's `gap_signal_multiplier` simultaneously (a) scaled
   the failure-logit coefficient AND (b) scaled `Gap_initial`'s own
   noise/variance upward. Both effects independently inflate apparent
   statistical power. `power_analysis_c1.py`'s own docstring — written
   at the repo's initial commit, ~7 weeks before `0d74cd4`, unmodified
   since — explicitly states `gap_signal_multiplier` and the trial-count
   `scale_factor` should be "varied **independently**," with the
   multiplier representing effect-size uncertainty alone. The pre-fix
   coupling contradicted this project's own older, documented design for
   what this parameter means.
4. **A real-world noise-magnitude comparison (requested to settle which
   version is "more realistic") could not be resolved** — the repo's
   only literature-calibrated noise reference (Nguyen et al. 2012,
   `retrain/scripts/noise_model.py`) calibrates raw depth-sensor noise
   in meters, a different physical quantity than the unitless
   visibility-fraction noise `mock_data_generator.py` actually uses. No
   valid comparison exists in this repo.

**Verdict**: the CURRENT (post-`0d74cd4`, decoupled) generator behavior
is the methodologically correct one, consistent with this project's own
earlier documented intent for this exact parameter. `PREREG.md`'s
"0.79-0.80 at 2x" fallback was not a considered, validated estimate of a
genuinely-more-optimistic-but-plausible scenario — it was an artifact of
a since-recognized coupling bug in the same code, and nobody re-checked
this specific PREREG claim after the bug was fixed for an unrelated
reason.

## Corrected power table

Reduced-scale reproduction (n_reps=100 per cell, consistent with
`project_audit/C6_AUDIT.md`'s methodology), CURRENT (decoupled, correct)
generator only — the ISOLATED/old-coupling column is included for
transparency about where PREREG's original number came from, not as an
alternative correct answer:

| N | mult | CURRENT (decoupled — use this) | OLD coupling (for reference only — NOT recommended) |
|---|---|---|---|
| 90 | 1x | 0.040 | 0.040 |
| 90 | 2x | 0.150 | 0.410 |
| 90 | 3x | 0.330 | 0.920 |
| 120 | 1x | 0.100 | 0.100 |
| 120 | 2x | 0.160 | 0.520 |
| 120 | 3x | 0.400 | 0.970 |
| 180 | 1x | 0.060 | 0.060 |
| 180 | 2x | 0.310 | 0.800 |
| 180 | 3x | 0.600 | 1.000 |
| 195 | 1x | 0.100 | 0.100 |
| 195 | 2x | 0.340 | 0.810 |
| 195 | 3x | 0.600 | 1.000 |

(Full commands and raw output:
`project_prereg_correction/scripts/full_power_table.py`, reproducible.)

**Corrected statement for PREREG.md's "2x" fallback claim**: under the
current, correctly-fixed generator, a 2x effect gives approximately
**0.31-0.34 power at N=180-195**, not 0.79-0.80. There is no multiplier
in {1x, 2x, 3x} at any tested N (90-195) that reaches the 0.80 target
under the current generator except 3x at N≥180 (0.60, still short) — no
tested cell reaches 0.80 at all under the current, correct model within
this range. `project_audit/C6_AUDIT.md` separately found (larger-scale,
1000/300-repeat runs, wider multiplier sweep) that even N=1440 (16x the
planned N) only reaches ~0.59 power at the nominal (1x) effect size.

## Remaining uncertainty

None on the central technical question (the two changes are separable;
the decoupled behavior matches this project's own older documented
intent) — this is not an ambiguous case requiring "both numbers as a
range." The genuine remaining uncertainty is the one this document
cannot resolve for anyone: **whether `b_gap=2.5` (or any multiple of it)
is anywhere near the REAL Gap-failure effect size**, since zero real
trials exist. That uncertainty exists regardless of which noise model is
used and is unaffected by this correction.

## Recommendation

**Keep `0d74cd4`'s noise-model change (the decoupling) — do not revert
it.** It is consistent with this project's own pre-existing, dated
documentation of what `gap_signal_multiplier` should mean, and the
pre-fix behavior's power numbers were inflated by a real, traced
mechanism (conflated effect-size and measurement-noise scaling), not by
a more-realistic noise assumption. **Apply this amendment to `PREREG.md`
by hand**, replacing the "approximately 0.79-0.80" figure with the
corrected ~0.31-0.34 at N=180-195, and updating the "Pre-registered
stance on this uncertainty" paragraph to remove the "meeting the target"
framing, since no plausible tested scenario (1x-3x, N=90-195) currently
meets it.
