# Overnight Log — overnight/prg-diagnosis

Session start: 2026-09-20 (autonomous, no human available until morning).
Branch created from `unified-pipeline` at commit `2a6f362` (added classifier files).

Rules in force: no hardware, no retraining/overwriting committed artifacts, no edits to `use_stub` in `integration/prg_config.json`, no in-place edits to PREREG.md (amendments only, in a separate file per task instructions), max ~3 attempts per problem then log and move on.

---

## [00:00] Setup

- Read PROJECT_STATUS.md, SCHEMA.md, INTERFACE.md, perception-calibration-gap/PREREG.md, integration/prg_config.json.
- Noted: `SCHEMA.md` and `INTERFACE.md` filenames have a trailing space in the actual filename (`SCHEMA.md `, `INTERFACE.md `) — not a typo in this log, an actual filesystem artifact. Using `cat "SCHEMA.md "` etc. to read them.
- Confirmed object-count conflict is actually three-way, not two-way as PROJECT_STATUS §5 item 4 suggested:
  - PREREG.md §5 (original, Day 2): 5 objects — apple, tennis_ball, purse, glass, remote.
  - PREREG.md Amendment 2026-07-15: still "5 physical objects" but remote→stapler swap, labeled "Option A confirmed."
  - INTERFACE.md §2 (reconstructed file): "Option A (agreed default): 3 physical objects (apple, purse, glass)."
  - This is a real, citable conflict for T5a — flagging, not resolving.
- Created branch `overnight/prg-diagnosis` from `unified-pipeline` @ 2a6f362.
- Committed PROJECT_STATUS.md (was untracked at session start).
- Config file baseline recorded for later diff check: `integration/prg_config.json` = `{ "use_stub": true }`.

Next: read relation_classifier/scripts/*.py before any experiment (T1 prerequisite).

---

## [T1a] Git archaeology — training_data_v3_metric_only.csv provenance

Commands run:
```
git log --all --stat -- '*training_data*'
git log --all --diff-filter=D --name-only | grep -i training
git log --all --diff-filter=A --name-only --pretty=format:"%H %ad" --date=short -- 'relation_classifier/data/*'
git log --all -p -- relation_classifier/scripts/extract_relation_features.py | grep -n PIPELINE_VALIDATION_MODE
git log --all --oneline -- relation_classifier/scripts/extract_relation_features.py
git show --stat e63cf944
cat .gitignore | grep -i csv
```
Findings:
- `relation_classifier/scripts/extract_relation_features.py` has exactly ONE commit in its entire history (`e63cf944`, 2026-07-30, "added the boundary depth estimation and relation classification pipeline"). `PIPELINE_VALIDATION_MODE = True` (i.e. `gazebo_range_gt`) in that commit and unchanged since — **the flag has never been False in this repo's git history.** No git evidence the training data was ever generated in `mde_predicted_z` mode.
- `logistic_model.joblib`, `scaler.joblib`, `models/summary.txt`, and `relation_classifier/data/training_data_metric_only.csv` (no "v3") all landed in that SAME single commit `e63cf944`.
- `models/summary.txt` line 1 says `Data source: data/training_data_v3_metric_only.csv` — the actual training run used a v3-named file that was **never once `git add`ed**, at any commit, on any branch (`git log --all --diff-filter=A -- 'relation_classifier/data/*'` returns only the one non-v3 file, ever).
- `.gitignore` (added in the same commit) explicitly excludes `relation_classifier/data/heuristic_vs_gazebo_comparison*.csv`, `flagged_objects.csv`, `*_gazebo_contacts.csv`, `data/_internal/`, `spot_check_sample.csv`, `pipeline_validation_*.csv` — i.e. every intermediate artifact `compare_heuristic_vs_gazebo_v3.py`'s pipeline produces on the way to a v3 table. `training_data_v3_metric_only.csv` itself isn't named in `.gitignore`, but it's a same-shaped generated file that fell through the same "generated CSVs aren't committed" policy.
- **Verdict: UNRECOVERABLE.** Not from git, not from any file on disk. Stating this plainly per task instructions.

## [T1b] Model self-consistency

Loaded `logistic_model.joblib` + `scaler.joblib` directly:
- `n_features_in_` = 8 (both), `classes_` = [0, 1], matches `FEATURE_COLUMNS` order in `classify_relation.py`.
- `coef_` = [-1.1077, 0.0950, 0.2183, -0.8461, 0.0990, 0.6129, 0.2145, -0.2290] (boundary_sharpness, minimum_boundary_distance, surface_normal_consistency, relative_height, depth_gradient, overlap_ratio, edge_continuity, occlusion_boundary_score), `intercept_` = 0.6566.
- These coefficients are **bit-identical** to the ones printed in the committed `models/summary.txt`, confirming this joblib is exactly the artifact from that one committed training run (not a later, different, unlogged retrain).
- `summary.txt` shows the model, ON ITS OWN reported held-out test split (64 rows) and 5-fold group CV, was NOT "always Separate": test recall on Contact = 0.857, 5-fold CV F1 mean = 0.803. **This rules out "the model itself is fundamentally degenerate/untrained" as an explanation** — it worked reasonably on its own native feature distribution.
- Could not run predict() on real training rows: the only recoverable CSV (`training_data_metric_only.csv`, non-v3, 277 rows) has only 3 of the 8 required features (`minimum_boundary_distance`, `overlap_ratio`, `relative_height`) — it predates the 8-feature v3 extraction and cannot be fed through this model. Noting this rather than fudging it.

## [T1c] Label-noise check

`relation_classifier/data/training_data_metric_only.csv`, `pd.crosstab(heuristic_label, gazebo_contact)`:
```
gazebo_contact    0   1
heuristic_label
Contact          60  41
Separate         88  13
Support           0  75
```
Exactly reproduces PROJECT_STATUS's "60 of 101 Contact calls" disagreement (60+41=101). Also newly noted: 13/101 heuristic "Separate" pairs are physically touching per `gazebo_contact` — the noise isn't one-directional. **Confirmed** as a real, independent contributing factor (this file predates v3 but is very likely drawn from the same authored scenes/heuristic logic that fed v3, given identical world/object naming and the same `SDFGeometryRelationSource`-style epsilon heuristic).

## [T1d] Real DA-V2 depth comparison — BLOCKED, logged per rule 8 (max 3 attempts)

Checked, cheapest-first:
1. Existing saved RGB/depth frames or per-world outputs in the repo: **none found** (`find` for `*.png` under any `world*` path, `*_depth.npy`: zero hits).
2. Headless Gazebo capture: `gz sim` binary IS present (v8.14.0) and `scripts/save_rgb_gz.py` / `scripts/depth_publisher.py` exist, but `depth_publisher.py` requires (a) `rclpy` — **not installed** in this Python environment (`ModuleNotFoundError: No module named 'rclpy'`), and (b) a DA-V2 checkpoint `.pth` file — **none exist anywhere in the repo** (`find . -iname "*.pth"` returns nothing; `checkpoints/` directory doesn't exist). Both are hard blockers independent of each other; fixing rclpy alone would not produce a usable depth prediction with no model weights.
3. Per task instructions: **logging the blocker and skipping T1d's direct DA-V2 comparison** rather than spending the night on ROS2/model-weight provisioning that needs a human decision (where do the checkpoint weights live / are they supposed to be downloaded from somewhere).

## [T1, new finding not in PROJECT_STATUS] Interface-region bug in feat_boundary_sharpness / feat_depth_gradient

Script: `diagnosis/scripts/run_binary_model_diagnostic.py` (committed). Ran the real, untouched `LiveDepthFeatureSource` → `classify_relation()` chain on 15 hand-picked Contact/Support/Separate pairs across all 6 worlds (same pairs `validate_live_depth_feature_source.py` already uses), via the same ray-cast renderer validated 182/0/18 last session. Reproduced PROJECT_STATUS's finding exactly: **15/15 predicted Separate**, p(Contact) < 2e-6 in every case, regardless of `SDFGeometryRelationSource` ground truth (5 Contact, 6 Support, 4 Separate labels among the 15).

z-scored every feature against the frozen `scaler.mean_`/`scale_`. `boundary_sharpness` and `depth_gradient` are 14.5–43.6 and 8.5–35.2 standard deviations, respectively, above the training-time mean in EVERY tested pair — an order of magnitude larger than any other feature's deviation. `boundary_sharpness` also carries the single largest-magnitude model coefficient (-1.1077).

Root cause, traced into the code (`extract_relation_features.py` lines 904-914, 988-997): `feat_boundary_sharpness()`'s "interface" region is `(ring_a | ring_b) & (dilate(mask_a) | dilate(mask_b))`, which is mathematically redundant (ring_a ⊆ mask_a ⊆ dilate(mask_a), so this reduces to just `ring_a ∪ ring_b`) — i.e. the ENTIRE silhouette ring of BOTH objects, not the shared A-B boundary. `feat_depth_gradient()` uses `mask_a | mask_b`, same issue. Measured directly on `world1_baseline`, `obj_bowl/obj_apple`: of 320 ring pixels, only 29 (9%) are on the true mutual A-B boundary; the rest face the background, which `ray_cast_scene()` sets to `camera.clip_far` (=2.0m in all 6 worlds; confirmed via direct inspection) for every non-hit pixel — a hard, un-anti-aliased ~1-1.5m jump at nearly every object's silhouette against 77-92% of the image being background. Restricting the same Sobel computation to ONLY the true mutual-boundary pixels drops `boundary_sharpness` from 3.05 (matches `LiveDepthFeatureSource`'s actual output almost exactly) to 0.57 — an order of magnitude, and far closer to the frozen model's training-time mean of 0.108. This pattern held across all 15 pairs / 6 worlds (script: `/tmp/.../check_boundary.py`, not committed — throwaway confirmation, superseded by the committed diagnostic script).

This fully, mechanistically explains why last session's depth-noise injection (5x, then 15x magnitude) never moved the verdict: the mismatch is systematic and ~15-40x too large in z-score terms for plausible sensor noise to close, and noise added near the object-object interface would not touch the dominant background-facing contamination this finding identifies.

**Important scope caveat**: this bug is in `feat_boundary_sharpness`/`feat_depth_gradient` themselves, which are the SAME functions used by the original `gazebo_range_gt` training-data extraction, `LiveDepthFeatureSource` (real/DA-V2 path), and this session's ray-cast validator alike — it is a property of the CODE, not specific to ray-cast depth. Whether it manifests as severely on real DA-V2 depth depends on how far real backgrounds are from foreground objects in real scenes (unknown, untestable tonight — see T1d blocker above).

## [T2] Corrected-feature rehearsal (SYNTHETIC WIRING REHEARSAL ONLY)

Script: `experiments/rerun_with_corrected_interface_SIM.py` (committed, not wired into any pipeline, does not modify `extract_relation_features.py`/`classify_relation.py`/any model or scaler). Recomputed ONLY `boundary_sharpness`/`depth_gradient` using the true mutual-boundary region instead of the codebase's ring_a|ring_b, left all 6 other features from the real `LiveDepthFeatureSource.compute()` untouched, fed the result through the real frozen `classify_relation()`.

Result: predictions went from 0/15 non-Separate (original) to 7/15 non-Separate (corrected) — genuine variation appeared where there was none. Binary (touching vs not) accuracy against `SDFGeometryRelationSource` ground truth improved from 4/15 (27%, "always guess Separate" baseline rate given 4 true-Separate pairs) to 9/15 (60%). Notably, all 5 SDF-labeled "Contact" pairs were STILL predicted Separate even after the fix — the fix helps Support-labeled pairs far more than Contact-labeled ones, which is consistent with (not proof of) the independently-confirmed T1c label-noise finding: SDF-heuristic "Contact" labels (the same style of static-pose heuristic used here as ground truth) disagree with physics-verified `gazebo_contact` 60/101 times, so some of these "Contact" ground-truth labels used for this comparison may themselves be wrong.

**Conclusion carried into the diagnosis doc**: the interface-region bug is a real, large, independently-confirmed contributor — fixing it alone moves the model from "degenerate" to "noisy but responsive" — but does not fully resolve the touching-model problem by itself; the label-noise issue (T1c) is very likely a second, independent, compounding factor, consistent with PROJECT_STATUS's original two-hypothesis framing, just with a different, code-provable first hypothesis than "MDE vs GT depth source."

