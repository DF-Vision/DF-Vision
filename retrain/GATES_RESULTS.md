# Gate Results — relation_binary_v2

Evaluated against the pre-committed gates in `retrain/GATES.md` (written
and committed before any training run). **Not softened after seeing
results.** Full raw outputs: `retrain/models/r3_training_summary.txt`
(CV), `retrain/r5_final_test_output_SIM.txt` (held-out world, run exactly
once), `retrain/scripts/r4b_fast_g5_check.py`'s printed output (this
document, below).

Selected candidate: `two_stage_logreg` (best of 6 tried; see
`retrain/models/r3_training_summary.txt`).

## G1 — Binary touching balanced accuracy > 0.75, bootstrap CI lower bound > 0.60

**CV (5-fold leave-one-world-out, 263 rows)**: mean = 0.6668, std =
0.0589, folds = [0.585, 0.676, 0.738, 0.615, 0.721]. **FAIL** (0.6668 <
0.75; no individual fold reaches 0.75 either).

**Held-out final test (world6_dense_mixed, 67 rows, run exactly once)**:
balanced accuracy = **0.8212**, bootstrap 95% CI (row-level, n_boot=2000)
= **[0.7256, 0.9020]**. **PASS** in isolation (0.8212 > 0.75, CI lower
bound 0.7256 > 0.60).

**Verdict: MIXED, reported honestly rather than picking whichever number
looks better.** The held-out world happened to score noticeably better
than the CV mean — plausibly because world6's actual pair mix (per
`retrain/data/MANIFEST.md`: 41 Support, 22 Separate, only 4 Contact) is
easier than some CV folds', not because the model generalizes better than
CV suggested. Gate condition explicitly requires holding on "leave-one-
world-out CV **plus** one fully held-out final test world" — read as a
conjunction, the CV failure alone is enough for **G1 overall: FAIL**, but
the discrepancy itself (and the resulting fold-to-fold instability, 0.585
to 0.738 in CV, then 0.821 held-out) is reported as a real finding, not
hidden.

## G2 — Each true class recall > 0.50, more than one class predicted

**CV**: recall(Separate)=0.6889 (n=135), recall(Contact)=**0.0000**
(n=37), recall(Support)=0.7802 (n=91). Consistent (Contact recall
0.00-0.27) across all 6 candidates tried, not specific to this one.

**Held-out**: recall(Separate)=0.9091 (n=22, PASS), recall(Contact)=**0.5000**
(n=4, exactly at the boundary — the gate requires strictly **>**0.50, so
this is a **FAIL by the letter of the stated threshold**, on an n=4
sample where the actual count is 2/4 correct), recall(Support)=0.7073
(n=41, PASS). 3 classes predicted (gate's ">1" condition satisfied).

**Verdict: FAIL**, though notably less decisively on the held-out set
(50% Contact recall, 2 of 4 examples right) than the CV numbers suggested
(0-27%). n=4 is too small to treat 50% as more than weak evidence either
way — reported as-is, not rounded up to a pass.

## G3 — Beats both dumb baselines by a margin whose CI excludes 0

**CV**: majority_class = 0.5000. best_single_feature_threshold = 0.6672.
Selected model (two_stage_logreg) = 0.6668. **FAIL** — the selected model
does not beat the single-feature-threshold baseline even on point
estimate (0.6668 < 0.6672); a margin-excludes-zero test is moot when the
point estimate isn't ahead to begin with. It does beat majority-class
(0.6668 vs 0.5000), but G3 requires beating both.

## G4 — ECE < 0.10, Brier score reported

**Held-out test**: Expected Calibration Error = **0.2278**, Brier score
= 0.1623. **FAIL** (0.2278 >> 0.10) — the model's confidence values are
poorly calibrated; a stated confidence does not reliably track actual
correctness rate on this held-out world.

## G5 — Balanced accuracy degrades < 0.10 absolute at noise=5x (or reported prominently)

The full 3-point [0,5,15] sweep (`retrain/scripts/r4_robustness_sweep.py`,
all 5 CV worlds) was started but killed after >12 minutes CPU on just the
first noise level, on a machine under variable heavy contention this
session (see `retrain/BASELINE.md`) — not completed, logged as a blocker
per rule 10 rather than left silently unresolved. A reduced, faster
substitute (`retrain/scripts/r4b_fast_g5_check.py`, 2 of the 5 CV worlds,
1 viewpoint each, n=12 pairs) completed:
```
balanced_acc @ noise=0: 0.8000
balanced_acc @ noise=5: 0.8500
degradation: -0.0500
```
**PASS on this reduced check** (|degradation| = 0.05 < 0.10) — but
**reported prominently, per the gate's own alternative clause, that this
is an n=12 sample and the direction (accuracy going UP with more noise)
is almost certainly sampling noise, not evidence of genuine robustness**.
This reduced check should not be read as a confident robustness
validation — it is a directional check under real time constraints, and
the honest caveat is that a proper 0-15x sweep across all 5 worlds was
not completed this session.

## G6 — prg_edge_error non-constant, in [0,1], AUROC > 0.65

**Held-out test**: n=67, range=[0.0591, 0.8519], 67 unique values (fully
non-constant), all in [0,1]: True. AUROC(prg_edge_error, wrong
prediction) = **0.5539**. **FAIL** (need >0.65) — `compute_prg_edge_error()`'s
output on v2 is well-formed (non-constant, correctly bounded) but not
meaningfully diagnostic of when v2's predictions are wrong; 0.5539 is
close to the 0.50 chance level.

## G7 — Tests pass, legacy regression identical, fresh-clone works, deterministic

**PASS.** `tests/test_relation_model_v2_regression.py`: 7/7 passing.
SHA256s of all legacy artifacts unchanged. Fresh-clone reproduction and
two-run determinism both confirmed identical. Full detail:
`retrain/R7_VERIFICATION.md`.

---

## Summary table

| Gate | Result | Verdict |
|---|---|---|
| G1 | CV 0.6668 (FAIL) / held-out 0.8212 (PASS) — mixed, CV governs | **FAIL** |
| G2 | CV Contact recall 0.00 / held-out 0.500 (boundary, n=4) | **FAIL** |
| G3 | Loses to single-feature-threshold baseline | **FAIL** |
| G4 | ECE=0.2278 | **FAIL** |
| G5 | Reduced check (n=12): degradation -0.05, within threshold, low confidence | **PASS (weak evidence)** |
| G6 | AUROC=0.5539 | **FAIL** |
| G7 | All sub-checks pass | **PASS** |

**5 of 7 gates fail.** Per the task's instruction ("If any gate fails,
the honest output is a failure report and a proposal. Do not soften
gates."): **v2 is not recommended for promotion.**
`relation_model_version` stays at its committed default, `"legacy"`.

## Interpretation

The interesting nuance this session surfaced, worth carrying forward: v2
performed noticeably BETTER on the one held-out world (0.821 balanced
accuracy, 50% Contact recall on a tiny n=4) than the 5-fold CV average
(0.667, 0% Contact recall) suggested it would. This kind of CV-vs-holdout
gap, combined with 5 of 7 gates failing, points toward a model that is
**unstable across scenes** rather than one that is uniformly bad — some
scene compositions it handles reasonably, others (per the CV folds) it
doesn't. This is a different, more specific characterization than either
prior session's framing ("always predicts Separate") or this session's
own CV-only view ("~67% ceiling") would suggest in isolation. A larger,
more diverse dataset (more than 330 rows, more than 6 scene templates)
would be needed to tell whether this instability is fundamental to the
feature set or just a small-sample-size artifact — see the recommendation
in `MORNING_HANDOFF_3.md`.
