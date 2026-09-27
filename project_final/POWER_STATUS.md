# Power Analysis Status — Consolidated

## Has `PREREG.md`'s amendment been applied?

**No — verified directly this session, not assumed.**
```
$ grep -n "0.79\|0.80\|0.31\|0.34" perception-calibration-gap/PREREG.md
65:power is approximately 0.12-0.15 -- well below the conventional 0.80
66:target. Achieving 0.80 power at this effect size would require
71:power at N=195 is approximately 0.79-0.80, meeting the target.
```
`PREREG.md`'s power-justification section still states the ORIGINAL,
now-known-to-be-stale "0.79-0.80 at 2x" figure. Its only `## Amendment`
sections are dated 2026-07-14/15 — nothing from any later session (the
structural-independence checkoff, Gap-only-primary, Model-D-conditional,
or this power correction) has been pasted in.

**MANASI: `project_prereg_correction/PREREG_AMENDMENT_DRAFT.md` still
needs to be applied by hand.** This session did not apply it (per the
absolute gate: `PREREG.md` is edited only via already-drafted amendment
files, and only by you).

## Corrected power table (from `project_prereg_correction/`, re-verified reproducible this session — see below)

| N | 1x (nominal) | 2x | 3x |
|---|---|---|---|
| 90 | 0.040 | 0.150 | 0.330 |
| 120 | 0.100 | 0.160 | 0.400 |
| 180 | 0.060 | 0.310 | 0.600 |
| 195 | 0.100 | 0.340 | 0.600 |

No cell in this table reaches the conventional 0.80 target. The original
`PREREG.md` figure (0.79-0.80 at N=195, 2x) was traced to a specific,
identified, since-fixed bug in `mock_data_generator.py` (commit `0d74cd4`)
that coupled the effect-size knob to the noise/measurement-spread of Gap
itself — a conflation this project's own OLDER documentation
(`power_analysis_c1.py`'s docstring, dated 7 weeks before the bug fix)
says should never have happened. The current, corrected numbers are the
methodologically correct ones — this is not an ambiguous finding needing
"both numbers as a range" (see `project_prereg_correction/COMMIT_DIAGNOSIS.md`
for the full mechanistic explanation and the reproduction that confirmed
this).

## The one real remaining uncertainty

**The true Gap-failure effect size is unknown, because zero real trials
exist.** `TRUE_COEFS["b_gap"] = 2.5` (the "1x" baseline every number
above is built on) is an arbitrary, hand-specified constant — its own
file's docstring calls the data it generates "FABRICATED" — with no
citation to a pilot dataset or literature value anywhere in this repo
(confirmed via repo-wide grep, `project_audit/C6_AUDIT.md`). This is not
something any amount of further simulation can resolve. The moment real
pilot trials exist, re-running `power_analysis_c1.py` with the
ACTUAL observed effect size (in place of `b_gap=2.5`) would replace every
number in the table above with a real one — that single step, not more
synthetic work, is what would move this forward.

## Reproducibility check (this session)

Re-ran the full corrected table directly from the committed
`project_prereg_correction/scripts/full_power_table.py` (unmodified, no
new experiment — a spot check that the file cited above is still
reproducible, not a new statistical claim):
```
$ python3 project_prereg_correction/scripts/full_power_table.py
    N  mult |  CURRENT (decoupled) | ISOLATED (null-fixed, coupled)
----------------------------------------------------------------------
   90   1.0 |                0.040 |                          0.040
   90   2.0 |                0.150 |                          0.410
   90   3.0 |                0.330 |                          0.920
  120   1.0 |                0.100 |                          0.100
  120   2.0 |                0.160 |                          0.520
  120   3.0 |                0.400 |                          0.970
  180   1.0 |                0.060 |                          0.060
  180   2.0 |                0.310 |                          0.800
  180   3.0 |                0.600 |                          1.000
  195   1.0 |                0.100 |                          0.100
  195   2.0 |                0.340 |                          0.810
  195   3.0 |                0.600 |                          1.000
```
Matches the table above exactly (same script, same seed, deterministic —
the "CURRENT (decoupled)" column is what's reported above; the "ISOLATED"
column is retained in the script for the mechanism explanation in
`COMMIT_DIAGNOSIS.md`, not a competing correct answer).
