# Project Status — FINAL (read this first, from now on)

Supersedes `PROJECT_STATUS.md`'s per-contribution sections (§1, §7-10) as
the first document to read for "what state is this project actually in."
`PROJECT_STATUS.md` itself is NOT deleted or edited — see the note
appended to it pointing here. This file exists because 8 sessions each
had their own partial status summary; this is the one that consolidates
all of them.

Contribution numbering is `PREREG.md`'s own (lines 9-30): **C1** =
Calibration Gap as an Independent Failure Predictor, **C2** = Post-hoc
Recalibration, **C3** = Gap × Occlusion × Clutter Interaction, **C4** =
Conformal Failure Gate, **C5** = Active Mitigation via TSDF Re-Fusion.
**C1-ext/C4-ext** = the dual-signal (Gap + PRG) extensions of C1/C4.
**Model D** = the combined two-signal model.

| Contribution | Status | One-line reason | Authoritative doc |
|---|---|---|---|
| **C1** (Gap as failure predictor) | **BLOCKED-ON-HARDWARE** (code fully built + synthetic-validated) | Every statistical component (`analysis_c1.py`, SHAP/fallback, per-object table) runs and is tested on SYNTHETIC data only; whether Gap actually predicts REAL failure needs real trials, which don't exist. Separately, even if it works, the study is currently underpowered at the assumed effect size (see `POWER_STATUS.md`). | `project_final/POWER_STATUS.md`, `perception-calibration-gap/analysis/analysis_c1.py` |
| **C2** (post-hoc recalibration) | **BLOCKED-ON-HARDWARE** (synthetic wiring check only) | `analysis_c2.py` runs on synthetic mock data and reports neither linear nor MLP correction beats the uncorrected baseline — but this is a SYNTHETIC result on fabricated data, not evidence about real recalibration; the real question needs real miscalibration to correct. | `perception-calibration-gap/analysis/analysis_c2.py` |
| **C3** (Gap×occlusion×clutter interaction) | **BLOCKED-ON-HARDWARE**, already pre-emptively downgraded | `PREREG.md`'s own power justification (lines 84-90) reclassifies C3 as exploratory, not confirmatory, BEFORE any data collection — estimated power 0.20-0.30 at the planned N, never approaching 0.80. `analysis_c3.py` on synthetic data: interaction term not significant (reported honestly as a negative result, per the pre-registered "do not drop the term" rule) — again, synthetic only. | `perception-calibration-gap/PREREG.md` lines 84-90, `analysis_c3.py` |
| **C4** (conformal failure gate) | **BLOCKED-ON-HARDWARE** (infrastructure fully built + synthetic-validated) | `conformal_gate.py`, `analysis_c4.py` (FAIL-Detect baseline, KS exchangeability, split-discipline test — `project/closure` C6 fixed a previously-vacuous split-discipline assertion) all run correctly and are tested on SYNTHETIC data; real coverage/efficiency needs real trials. | `perception-calibration-gap/mock-pipeline/conformal_gate.py`, `tests/test_split_discipline.py` |
| **C5** (TSDF re-fusion active mitigation) | **BLOCKED-ON-HARDWARE** (synthetic wiring check only) | `analysis_c5.py` on synthetic data: recovery rate is exactly 0/2, reported honestly per its own pre-committed criterion ("recovery_rate > 0... FAIL"). Synthetic-only result — not evidence the real mitigation mechanism doesn't work, just that this session's fabricated scenario didn't show it. | `perception-calibration-gap/analysis/analysis_c5.py` |
| **C1-ext / C4-ext** (dual-signal Gap+PRG extensions) | **DEAD-END-CONFIRMED** (for every design tried; real-depth testing remains technically untried) | 5 sessions, 3 distinct design approaches (trained classifier, ring-based classifier-free ×3, depth-only connected-components classifier-free), ALL failed decisively against pre-committed, never-loosened gates — on SIMULATION data alone, before hardware would even be reached. The one thing that COULD still change this (real DA-V2/camera depth, never tested) is laid out precisely in `PRG_SIGNAL_STATUS.md`'s closing paragraph and `MASTER_HARDWARE_CHECKLIST.md`'s Track C — but every existing design is confirmed dead in its current form. | `project_final/PRG_SIGNAL_STATUS.md` (full 7-point history) |
| **Model D** (combined two-signal model) | **BLOCKED** (structurally, by design, on both C1-ext/C4-ext AND real trial data) | Fully built and tested (`model_d.py`, 11/11 tests) but refuses to run unless explicitly enabled AND fed real (non-stub) PRG rows — neither condition is met, and `use_stub=true` stays true (this consolidation does not change that, per the absolute gate). Not reachable until C1-ext/C4-ext produces a working signal, which per the row above is currently a confirmed dead end. | `perception-calibration-gap/analysis/model_d.py`, `project_final/PRG_SIGNAL_STATUS.md` |

## Addendum — 2026-09-27 (`project/closure-actions`)

**Decision (2026-09-27, Manasi/Krupa): PRG dual-signal extension is
formally retired for this paper submission regardless of Track C's
outcome. Gap-only (C1, C4) is the confirmed sole primary path. Track C's
real-depth capture, if and when run, is documented as
confirmatory/appendix material only.** This does not change the C1-ext/
C4-ext or Model D rows above (they remain accurate technical
descriptions of what was tried and why it's dead/blocked) — it resolves
the SCOPING question the table's own text left open (`PRG_SIGNAL_STATUS.md`'s
prior closing paragraph, and `RESOLUTION_LOG.md` row 11). Decision also
recorded in `perception-calibration-gap/PREREG.md`'s
`## Amendment 2026-09-27 — Power analysis correction` section and in
`PRG_SIGNAL_STATUS.md`'s closing paragraph.

## What "BLOCKED-ON-HARDWARE" means precisely, for every C1-C5 row above

The CODE is done: built, tested, mutation-hardened where a guard exists
(`project_final/MUTATION_STATUS.md`), and runs correctly end-to-end on
synthetic data (`perception-calibration-gap/e2e_integration_rehearsal_SYNTHETIC.py`).
What's missing is not more software — it's real trial data. No further
autonomous session can change this row's status; only real hardware use
(`project_final/MASTER_HARDWARE_CHECKLIST.md`) can.

## What "DEAD-END-CONFIRMED" means precisely, for the C1-ext/C4-ext row

This is NOT the same as "needs hardware to know." These designs already
failed on the EASIER test (simulation, idealized/synthetic scenes) that
would need to pass BEFORE real-hardware testing would even be worth
attempting. The diagnosed failure mechanism (direction-ambiguous
background contamination in any single-target-bbox sampling scheme) has
now defeated two independent mitigation attempts. A fourth design attempt
is not something any further autonomous session should undertake without
a genuinely new idea — see `PRG_SIGNAL_STATUS.md`'s closing paragraph.
