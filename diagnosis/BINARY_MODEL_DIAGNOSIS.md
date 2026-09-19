# Binary Touching-Model Diagnosis

Overnight autonomous session, 2026-09-20. Branch `overnight/prg-diagnosis`.
Answers PROJECT_STATUS.md §1: why does `relation_classifier/models/logistic_model.joblib`
classify every tested Contact/Support pair as "Separate" at ~100% confidence?

**Scope discipline**: every number in this document derived from ray-cast
synthetic depth on staged SDF scenes is labeled SIM/SYNTHETIC and is not
citable evidence for C1/C1-ext/C4/C4-ext. No model, scaler, or committed
training CSV was modified, retrained, or overwritten to produce this
document. `integration/prg_config.json` was not touched (`git diff` clean,
confirmed after every step, see OVERNIGHT_LOG.md).

All exact commands and raw output are in `OVERNIGHT_LOG.md`; this document
states verdicts and reasoning. Reproduce any finding here with:
```
python3 diagnosis/scripts/run_binary_model_diagnostic.py
python3 experiments/rerun_with_corrected_interface_SIM.py
```

---

## Hypotheses tested

### H1 — Model trained on `mde_predicted_z` depth, tested on `gazebo_range_gt`-style depth (PROJECT_STATUS's leading hypothesis)

**Verdict: INCONCLUSIVE — could not be directly tested; weakened by available evidence.**

- Direct test (feed real DA-V2 depth through the same chain) is **BLOCKED**: no DA-V2 checkpoint (`.pth`) file exists anywhere in this repository, and `rclpy` (required by `scripts/depth_publisher.py`) is not installed in this environment. Both are independent hard blockers; this is not something an overnight software-only session can provision safely (checkpoint provenance/download source is a decision for a human). Logged per the task's max-3-attempts rule rather than burning the night on ROS/model-weight setup with no more evidence it would even resolve the question.
- Indirect evidence against: `relation_classifier/scripts/extract_relation_features.py` has exactly **one commit in its entire git history** (`e63cf944`, 2026-07-30), and `PIPELINE_VALIDATION_MODE = True` (→ `gazebo_range_gt`) in that commit, unchanged since. There is no git evidence the code was ever run in `mde_predicted_z` mode to produce training data. This doesn't prove it wasn't (someone could have locally toggled the flag, generated data, and reverted before committing — untraceable), but it removes the only affirmative evidence for H1 and means H1 currently rests on inference about file *paths* (`training_data_v3_metric_only.csv` vs. `training_data_metric_only.csv`) rather than any confirmed code-path difference.
- **This hypothesis should not be treated as confirmed or as the primary fix target** based on what could be checked tonight. See H3 below for a hypothesis that IS confirmed and does not depend on DA-V2 at all.

### H2 — Training labels are noisy (`heuristic_label` vs. physics-verified `gazebo_contact` disagree)

**Verdict: CONFIRMED** (already on record per PROJECT_STATUS; independently reproduced this session).

`relation_classifier/data/training_data_metric_only.csv` (the one recoverable data file, 277 rows):
```
gazebo_contact    0   1
heuristic_label
Contact          60  41
Separate         88  13
Support           0  75
```
60/101 heuristic "Contact" calls are NOT physics-verified contact (matches PROJECT_STATUS exactly). New in this pass: 13/101 heuristic "Separate" calls ARE physics-verified contact — the noise is not one-directional. A model trained toward `gazebo_contact` using features whose "Contact" support in the SDF-authored scenes is only 41% physics-clean would plausibly learn a distorted, conservative boundary independent of any depth-source question. This file is a different (larger, fewer-feature) dataset than the 189-row v3 file the deployed model was actually fit on, but shares the same authoring/heuristic pipeline and world set, so this finding is treated as a strong, load-bearing prior about v3's label quality too — not proof about v3 specifically.

### H3 — `feat_boundary_sharpness()` / `feat_depth_gradient()`'s "interface" region is dominated by background, not the true object-object boundary (new hypothesis, not in PROJECT_STATUS)

**Verdict: CONFIRMED, with a quantified, code-level mechanism.**

Reproduced the "always Separate" finding fresh (`diagnosis/scripts/run_binary_model_diagnostic.py`): 15/15 staged pairs across all 6 worlds predict "Separate" with p(Contact) < 2e-6, regardless of ground-truth relation.

Z-scoring every feature against the frozen `scaler.mean_`/`scale_` shows `boundary_sharpness` at **14.5–43.6 standard deviations** above the training-time mean in every single pair tested — an order of magnitude beyond any other feature's deviation — and `depth_gradient` at 8.5–35.2 SDs. `boundary_sharpness` also carries the model's single largest-magnitude coefficient (-1.1077 of 8, next-largest is `relative_height` at -0.8461). A z-score in the 15-40 range on a -1.1 coefficient alone contributes roughly -17 to -44 to the logit, which saturates `predict_proba` to ~0 for Contact regardless of what any other feature says.

Traced to source (`extract_relation_features.py:904-914` and `:988-997`): the "interface" region used by both features is defined as (effectively, after removing a redundant AND) `ring_a ∪ ring_b` — the full silhouette ring of BOTH objects, not the shared A-B boundary. Measured directly on `world1_baseline`, `obj_bowl`/`obj_apple`: of 320 ring pixels, only 29 (9%) lie on the true mutual boundary; the rest border the background. `ray_cast_scene()`'s background convention (deliberately matching what it believes real depth sensors do — see its own docstring) sets every non-hit pixel to `camera.clip_far` (2.0m in all 6 worlds), producing a hard, zero-anti-aliasing ~1-1.5m jump at 77-92% of each object's silhouette. Restricting the identical Sobel computation to only the true mutual-boundary pixels drops `boundary_sharpness` from 3.05 (essentially exactly what `LiveDepthFeatureSource` returns) to 0.57 on this pair — nearly matching the frozen model's training-scale mean of 0.108, versus the raw feature's 3.05. This pattern held on every one of the 15 tested pairs (see OVERNIGHT_LOG.md for the full per-pair table).

This mechanistically explains why last session's depth-noise injection (up to 15x the literature-grounded magnitude) never moved the verdict "to six decimal places": the discrepancy is systematic and 15-40x too large in z-score terms for plausible sensor noise — noise added near the true interface wouldn't touch the dominant background-facing contamination this finding identifies, and noise scaled to realistic sensor levels can't close a multi-standard-deviation systematic offset.

**Confirmatory experiment** (`experiments/rerun_with_corrected_interface_SIM.py`, SIM-labeled, does not modify any production file/model/scaler): recomputing only these two features on the true mutual-boundary region, leaving the other 6 features untouched, and feeding the result through the real frozen classifier moved predictions from **0/15 non-Separate → 7/15 non-Separate** — genuine variation appeared. Binary (touching-vs-not) accuracy against SDF-geometry ground truth rose from 4/15 (27%) to 9/15 (60%). Notably, all 5 SDF-labeled "Contact" pairs were STILL predicted Separate after the fix — consistent with H2: the "Contact" ground-truth labels used for this comparison are the same style of static-pose heuristic that H2 shows disagrees with physics-verified contact 60/101 times, so some of this comparison's own "Contact" labels may themselves be wrong, not just the model.

**Scope caveat, important**: `feat_boundary_sharpness`/`feat_depth_gradient` are the SAME functions used by the original `gazebo_range_gt` training extraction, by `LiveDepthFeatureSource` on real/DA-V2 depth, and by this session's ray-cast validator — the bug is in the feature code, not specific to ray-cast depth. Whether it manifests this severely on real DA-V2 depth depends on how far real backgrounds sit from foreground objects in real scenes, which is untestable tonight (H1's blocker applies here too).

---

## Combined picture

The "always Separate" behavior is best explained by **H3 (confirmed, large, primary) compounding with H2 (confirmed, independent, secondary)** — not by H1 (untested, weakly evidenced). Fixing H3 alone (per the rehearsal) moves the model from *completely degenerate* (0 variation, wrong on everything but true-Separate) to *noisy but responsive* (60% binary accuracy on a 15-pair synthetic sample, all improvement concentrated in Support cases, none in Contact cases) — which is exactly the pattern you'd expect if a second issue (label noise on the Contact class specifically, H2) is still capping accuracy after the larger bug is fixed.

## Recommendation

1. **Do not retrain tonight** (rule 3) — this is a proposal only, for morning review.
2. **The interface-region bug (H3) is a `extract_relation_features.py` code fix, not a retraining decision**, and is owned by Person B (Krupa). It affects `feat_boundary_sharpness` and `feat_depth_gradient` for ALL callers (training extraction, `LiveDepthFeatureSource`, the ray-cast validator) — fixing it changes the FEATURE DEFINITION, which means:
   - The frozen `logistic_model.joblib`/`scaler.joblib` were fit on the OLD (buggy) feature definition. Fixing the feature extractor without retraining would put every future feature vector out of distribution relative to what the frozen scaler/model expect, in the opposite direction from today's bug. **A feature-definition fix and a model retrain must ship together, not separately** — this is the one piece of this diagnosis that does point toward retraining, but as a *consequence of a feature-code fix*, not a standalone "the model is bad" retrain.
   - Suggested minimal fix (for Krupa to review, not applied here): redefine the "interface" region in both functions to the true mutual boundary — e.g. `(ring_a & dilate(mask_b)) | (ring_b & dilate(mask_a))`, which is exactly what `experiments/rerun_with_corrected_interface_SIM.py` used for the rehearsal above.
3. **The label-noise issue (H2) needs a scoping decision, not silent action** (per PROJECT_STATUS's own prior framing, reaffirmed here): whether to re-derive `heuristic_label`-based "Contact" training rows against `gazebo_contact` before any retrain, weight `gazebo_contact` more heavily, or drop disagreeing rows. This is a bigger, more consequential change and should wait for you.
4. **H1 (MDE-vs-GT) should be tested for real before being ruled in or out** — once a DA-V2 checkpoint and an `rclpy` environment are available, re-run `diagnosis/scripts/run_binary_model_diagnostic.py`'s chain with real DA-V2 depth on the same staged scenes (swap `ray_cast_scene()`'s depth for `depth_publisher.py`'s output) and check whether `boundary_sharpness`/`depth_gradient` land closer to the training-time mean on real sensor depth than on ray-cast depth — that would separately confirm or rule out whether H1 is ALSO a contributing factor beyond H3.
5. **Safe to test/deploy against, right now, with the frozen model, with no code changes**: nothing. Given H3 is confirmed and unfixed, no depth source (GT-style or MDE-style) can be expected to produce sensible predictions from the current frozen model + current feature code, because the bug is in the feature computation, not the depth source. This directly informs T2 below.
