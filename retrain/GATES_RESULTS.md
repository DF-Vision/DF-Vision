# Gate Results — relation_binary_v2

Evaluated against the pre-committed gates in `retrain/GATES.md` (written
and committed before any training run, per rule/instruction). **Do not
soften these thresholds** — this document reports actual numbers and a
PASS/FAIL against the gates as written, not an adjusted judgment call.

Selected candidate (per `retrain/models/r3_training_summary.txt`):
`two_stage_logreg` (logistic regression binary touching model + logistic
regression Support-vs-Contact sub-classifier, matching the legacy
architecture) — this WAS the best of 6 candidates tried (2 architectures
x 3 algorithms) on leave-one-world-out CV over `world1_baseline`,
`world2_dense_clutter`, `world3_thin_objects`, `world4_support_scene`,
`world5_occlusion_scene` (263 rows). `world6_dense_mixed` (67 rows) was
never touched during selection.

## G1 — Binary touching balanced accuracy > 0.75, bootstrap CI lower bound > 0.60

**CV (5-fold leave-one-world-out)**: mean = 0.6668, std = 0.0589, folds =
[0.585, 0.676, 0.738, 0.615, 0.721].

**FAIL.** 0.6668 < 0.75. This is not a marginal miss — every one of the 5
folds individually is below 0.75 as well (best fold: 0.738).

**Final held-out test (world6_dense_mixed, R5)**: see
`retrain/r5_final_test_output_SIM.txt` for the literal run output —
[PLACEHOLDER: filled in after R5 runs, per rule 8's "touch the test set
exactly once" — this document is updated with R5's numbers immediately
after that one run, not before].

## G2 — Each true class recall > 0.50, more than one class predicted

**CV**: recall(Separate)=0.6889 (n=135), recall(Contact)=**0.0000**
(n=37), recall(Support)=0.7802 (n=91).

**FAIL, decisively, and not specific to this one candidate.** Contact
recall was 0.00-0.27 across ALL 6 candidates tried (two-stage and
direct-3class, logistic regression / random forest / gradient boosting
alike) — see `retrain/models/r3_training_summary.txt`. This points at a
structural limitation of the 8-feature set / label quality for
distinguishing "Contact" (side-by-side touching) specifically, not a
single-model failure fixable by trying another algorithm.

## G3 — Beats both dumb baselines by a margin whose CI excludes 0

**CV**: majority_class = 0.5000 (std 0). best_single_feature_threshold =
0.6672 (std 0.0869). Selected model (two_stage_logreg) = 0.6668 (std
0.0589).

**FAIL against the single-feature-threshold baseline** — the selected
model does not even numerically beat it on mean CV score (0.6668 <
0.6672); a margin-excludes-zero test is moot when the point estimate
itself isn't ahead. It does beat the majority-class baseline (0.6668 vs
0.5000), but G3 requires beating BOTH baselines, so this gate fails as a
whole.

## G4 — ECE < 0.10, Brier score reported

[PLACEHOLDER: computed in R5's held-out test, filled in after that run —
see note under G1.]

## G5 — Balanced accuracy degrades < 0.10 absolute at noise=5x, or reported prominently

[PLACEHOLDER: `retrain/scripts/r4_robustness_sweep.py` sweep in progress
at the time this section was drafted — reduced to noise levels [0, 5, 15]
after the full 7-point sweep proved too slow under this session's CPU
contention (see that script's own comment) for the marginal value it
would add given G1/G2 already fail decisively. Filled in below once the
run completes.]

## G6 — prg_edge_error non-constant, in [0,1], AUROC > 0.65

[PLACEHOLDER: computed in R5's held-out test.]

## G7 — Tests pass, legacy regression identical, fresh-clone works, deterministic

`tests/test_relation_model_v2_regression.py`: 7/7 passing (confirmed this
session, see `OVERNIGHT_LOG.md`). Fresh-clone and determinism checks in
`retrain/R7_VERIFICATION.md`.

---

## Interim conclusion (before G4-G7 are filled in)

**G1, G2, and G3 already fail, decisively rather than marginally, using
clean, provenanced, real-physics-verified training data and the exact
production feature-extraction code (train/serve parity enforced by
construction).** This is not a data-quality problem in the sense the
prior two sessions worried about (label noise, lost provenance, ring-
kernel mismatches) — those were all fixed or accounted for in this
session's `retrain/data/relation_dataset_v2_SIM.csv`, and the result is
still a hard ceiling around 65-67% balanced accuracy with near-zero
Contact-class recall, consistent across 6 different model
architecture/algorithm combinations. **The most likely remaining
explanation is that the 8-feature set itself does not carry enough
signal to reliably separate side-by-side Contact from stacked Support or
non-touching, at least on ray-cast synthetic depth** — see the
recommendation in `MORNING_HANDOFF_3.md` for what this implies.

Per the task's explicit instruction ("If any gate fails, the honest
output is a failure report and a proposal. Do not soften gates."): **v2
is not recommended for promotion to the default.** `relation_model_version`
stays at its committed default, `"legacy"`.
