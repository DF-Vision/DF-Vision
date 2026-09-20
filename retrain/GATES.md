# Pre-committed Acceptance Gates — relation_binary_v2

Written and committed BEFORE any training run. These will not be changed
after seeing results. If any gate fails, the honest output is a failure
report and a proposal, not a softened threshold.

## Gates (verbatim from task spec)

The v2 model may be recommended only if ALL hold, on data it never
trained or tuned on (leave-one-world-out CV plus one fully held-out final
test world):

- **G1**: Binary (touching vs not) balanced accuracy > 0.75, with
  bootstrap 95% CI lower bound > 0.60 (bootstrap grouped by world/scene).
- **G2**: Recall of each true class (Separate, Contact, Support) > 0.50,
  and more than one class predicted.
- **G3**: Beats both dumb baselines (majority class, best single-feature
  threshold) by a margin whose CI excludes 0.
- **G4**: Calibration: Expected Calibration Error < 0.10 and Brier score
  reported (prg_edge_error consumes the confidence).
- **G5**: Robustness: balanced accuracy under the noise sweep (R4)
  degrades by less than 0.10 absolute at the highest realistic noise
  level, OR the degradation is reported prominently (i.e. this gate can
  be satisfied by disclosure, not just by passing — reported either way).
- **G6**: `compute_prg_edge_error()` on v2 outputs is non-constant, in
  [0,1], and AUROC > 0.65 for flagging wrong predictions.
- **G7**: All tests pass, legacy regression identical, fresh-clone
  reproduction works, two seeded runs give identical outputs.

## Pre-committed decisions that affect how gates are evaluated (fixed now, before data generation)

1. **Held-out final test world**: `world6_dense_mixed`. Rule, decided now:
   pick the numerically-last, most-heterogeneous-by-name world
   ("dense_mixed" implies the broadest mix of object/relation types of
   any of the 6, making it the hardest single-world stress test available)
   — not chosen by looking at any result, class balance, or model
   performance. This world is touched exactly once, at R5, after all
   model selection is finished on the other 5.
2. **CV structure**: leave-one-world-out (5-fold, over
   `{world1_baseline, world2_dense_clutter, world3_thin_objects,
   world4_support_scene, world5_occlusion_scene}` — `world6_dense_mixed`
   excluded entirely from this phase). Hyperparameters tuned only inside
   training folds.
3. **Random seeds**: `RANDOM_STATE = 20260920` (today's date, arbitrary
   but fixed and logged) for every stochastic step (data noise sampling,
   sklearn model fitting, bootstrap resampling). Logged again at each use
   site in `OVERNIGHT_LOG.md`.
4. **Bootstrap**: 2000 resamples, grouped by world (resample the set of
   worlds with replacement, not individual rows), 95% CI = [2.5, 97.5]
   percentile, matching the methodology already used in
   `verification/scripts/v1f_rerun_with_real_gazebo_contact.py`.
5. **Baselines for G3**: (a) majority-class predictor (predicts whatever
   class is most common in the training folds); (b) best single-feature
   threshold, selected by trying every one of the 8 features and every
   threshold at the training folds' unique-value midpoints, evaluated the
   same way as the candidate model (grouped CV), reporting whichever
   feature/threshold combination scored highest — matching the standard
   this project already used for the Support sub-classifier's threshold
   baseline (`relation_classifier/models/support_summary.txt`).
6. **"Realistic noise level" for G5**: the sweep in R4 goes from 0x to
   15x a documented base noise magnitude (matching the range last used in
   the pre-this-session diagnosis). "Highest realistic level" for G5's
   specific threshold is defined now as **5x base** (not 15x) — 15x was
   explicitly a last-session STRESS test to prove noise alone couldn't
   explain the original bug, not a claim that 15x reflects a plausible
   real sensor; 5x is the more defensible upper bound of "realistic"
   sensor variation for G5's specific pass/fail purpose. The full 0-15x
   curve is still reported for transparency regardless of which point G5
   is evaluated at.
7. **Train/serve parity (rule 7, restated as it bears on gate evaluation)**:
   every feature used in training, CV, the held-out test, and the noise
   sweep is produced by calling the exact same `LiveDepthFeatureSource`
   class (or, for v2, whatever single shared feature-producing function
   both training and `classify_relation.py`'s v2 path call) — never a
   separate "training-time" feature computation. If this constraint ever
   forces a design choice, that choice is logged here or in
   `OVERNIGHT_LOG.md` at the point it's made.

## Not-yet-known parameters, to be filled in as data is generated (not results)

- Exact candidate model list (logistic regression, random forest, small
  gradient boosting, plus direct-3-class vs two-stage) — fixed in R3
  before looking at CV scores, per standard nested-CV discipline.
- Class-weighting scheme for imbalance — decided from the training data's
  actual class counts once generated (R2), not tuned against CV score.
