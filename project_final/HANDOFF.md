# F9 — Final Handoff, `project/final-integration`

Branch `project/final-integration`, created from `project/prereg-correction`
(the most complete verified tip). Not merged into `unified-pipeline` or
anywhere else — that decision is explicitly left to Manasi, per the
task's stop conditions. 8 commits (F0-F8), all forward-only, reviewed in
`project_final/VERIFICATION.md`.

## Is the software side "done"?

**Yes, in the sense that every planned analysis path (C1-C5, C1-ext/
C4-ext, Model D) is built, tested (146 passing tests), and — everywhere
a safety- or statistics-critical guard exists — mutation-hardened (30/30
kill rate, re-verified this session with zero regression). The full
pipeline runs end-to-end on synthetic data with zero errors
(`e2e_integration_rehearsal_SYNTHETIC.py`), every prior session's flagged
loose end is either resolved or explicitly logged as open with a reason
(`RESOLUTION_LOG.md`), and this session found and root-caused one
previously-unnoticed pre-existing test-script bug along the way
(`TEST_CONSOLIDATION.md`) without needing to write new code to do so.**

**No, in the sense that zero real trials exist, and several central
questions cannot be answered by any amount of further software work**:
does Gap actually predict failure at the real effect size (unknown —
`b_gap=2.5` is an arbitrary placeholder, `POWER_STATUS.md`); does any PRG
signal work on real depth (untested — every existing design already
failed on the EASIER simulation-only test, `PRG_SIGNAL_STATUS.md`); does
the calibrated rig, lookup table, and conformal gate behave as designed
on a physical arm (untested — zero hardware sessions have run).
**Pretending either of these gaps is closed would be the one true way to
fail this consolidation** — they are not closed, and this document does
not claim otherwise.

## `RESOLUTION_LOG.md` summary

14 distinct flags found across all 8 prior sessions' docs. **6 resolved**
with passing tests/mutation evidence (shap fix, conformal-coverage
single-seed anomaly, `hand_calc_validation.py` template guard, the
`RETRAIN_OR_RELABEL_PROPOSAL.md` thread, plus 2 cross-references). **5
are fully-drafted documentation amendments never applied** to `PREREG.md`/
`SCHEMA.md ` — zero code impact, waiting only on your manual paste
(see next section). **2 are genuine open decisions** only you can make:
the object-count conflict (5 vs. 5-with-swap vs. 3, still live in both
`PREREG.md` and `INTERFACE.md`, never resolved across 4+ sessions) and
whether to keep attempting new PRG signal designs or formally retire that
contribution. **1 is an infrastructure/logistics question** (where should
the DA-V2 checkpoint weights live — committed, symlinked, or documented
as an external dependency; a checkpoint exists outside the repo at
`/home/manasi/bps-project/checkpoints/` but no session has decided what
to do about that). The branch-merge-decision flag that opened this whole
investigation (`a/overnight-hardening`, `origin/failure-calibration`) was
resolved this session — both non-ancestor commits inspected and rejected
with logged reasons, neither cherry-picked.

## Does `PREREG.md` need manual action?

**Yes — confirmed this session by direct read, not assumed.** `PREREG.md`'s
power-justification section still states the stale "0.79-0.80 at 2x"
figure; its only `## Amendment` sections are dated 2026-07-14/15. **Apply
`project_prereg_correction/PREREG_AMENDMENT_DRAFT.md` by hand** —
corrected figure ~0.31-0.34, full reasoning already drafted. Also still
unapplied: the structural-independence checkoff and the Gap-only-primary/
Model-D-conditional/PRG-history amendments in `PREREG_AMENDMENTS_PROPOSED.md`
(9 items total), and `SCHEMA.md `'s field-addition amendments
(`SCHEMA_AMENDMENT_PROPOSED.md`) plus the `training_data_v3_metric_only.csv`
provenance note (`diagnosis/DATA_PROVENANCE_NOTE_PROPOSED.md`). None of
these need further drafting — they're ready, reviewed, and waiting on
you specifically because this project's own rule (and this session's
hard gate) is that only you edit those 3 frozen files.

## Master hardware checklist

**`project_final/MASTER_HARDWARE_CHECKLIST.md`** — merges Manasi's
9-step calibration/arm/YOLO protocol with Krupa's real-depth PRG capture
protocol into one ordered document with 4 explicit tracks, including the
one genuinely parallelizable fact worth acting on immediately: **Krupa's
real-depth capture (Track C) needs only the camera and printed/measured
markers — it does not need to wait for any of Manasi's arm-calibration
work**, and it is the single highest-value next hardware step, since it's
the only thing that could actually change `PRG_SIGNAL_STATUS.md`'s
current verdict.

## No further autonomous overnight session should be run on this repo's software until real trial or real-capture data exists to analyze

Every remaining open question in this project is now empirical, not
computational. The statistical machinery, the safety guards, the
calibration solver, the conformal gate, the YOLO pipeline, the PRG
candidates (three distinct designs now) — all exist, are tested, and are
mutation-hardened where it matters. Running another overnight session
against the current repo would, at best, re-verify what
`project_final/VERIFICATION.md` already re-verified, and at worst
produce a fourth synthetic PRG design attempt or another simulated power
analysis — neither of which this consolidation's own evidence suggests
would move anything forward. **The next productive session is a hardware
session** (`MASTER_HARDWARE_CHECKLIST.md`), not another autonomous
software pass. If a future autonomous session IS run anyway, its first
job should be reading this file and `PROJECT_STATUS_FINAL.md` in full
before touching anything.
