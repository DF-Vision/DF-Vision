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

Wrote up full T1e diagnosis in `diagnosis/BINARY_MODEL_DIAGNOSIS.md` (hypotheses H1/H2/H3, verdicts, recommendation). Committed `7cb1eba` (scripts), `3c8deba` (diagnosis doc).

## [T3] Wider prg_edge_error sweep

Script: `diagnosis/scripts/sweep_prg_edge_error_SIM.py` (committed). Swept ALL adjacent pairs (via `enumerate_adjacent_pairs`, not just the 15 hand-picked ones) across all 6 worlds: 43 pairs total. Ground truth: `SDFGeometryRelationSource` (static-pose heuristic — the only ground truth available without a running physics sim; explicitly caveated in the script and output doc as not physics-verified, same caveat as T1's use of it).

Results (`diagnosis/PRG_EDGE_ERROR_SWEEP_SIM.md`, `diagnosis/prg_edge_error_sweep_SIM.csv`):
- prg_edge_error by gt_relation: Contact mean=0.194, Separate mean=0.111, Support mean=0.281 (all in [0,1], only 4 unique values across the sweep — expected, since `compute_prg_edge_error`'s confidence_term barely varies while the classifier is degenerate per T1).
- Binary wrong-prediction rate: 79.1% (34/43) — consistent with T1's finding that the model is currently near-always-wrong on touching pairs.
- AUROC(prg_edge_error, wrong-prediction) = 0.62, point-biserial = 0.20 — a real but weak signal.
- Interpretation recorded in the doc: given the classifier's confirmed near-degeneracy (H3), this AUROC likely reflects `compute_prg_edge_error()`'s geometric_term (a raw depth-gap check) acting almost independently of classifier confidence, not the two-signal design working as intended. Should be re-run once H3 is fixed before drawing real conclusions.

Committed `2593334`.

## [T4] Model D + two-signal conformal gate — synthetic wiring rehearsal

Checked for Person A's stats code first: `perception-calibration-gap/mock-pipeline/conformal_gate.py` and `gap_computer.py` exist (single-signal `ConformalGate` class, Gap-only nonconformity score) — proceeding was in scope per task instructions.

Script: `rehearsal/model_d_and_two_signal_gate_REHEARSAL.py` (committed, new file, does not modify `conformal_gate.py`/`gap_computer.py`). Built a 43-row synthetic trial table (1 row per T3's real prg_edge_error value; Gap_initial/true_occlusion/failure all synthetic with a deliberately planted logit relationship, purely so Model D has something non-trivial to recover). Fit Model D (`failure ~ true_occlusion + Gap_initial + prg_edge_error + C(object_id)`, statsmodels logit) and a max-combination two-signal conformal gate (Gap and prg_edge_error each scaled by their own calibration-set 90th percentile, combined via max()), alongside the existing single-signal `ConformalGate` for comparison.

Note: first attempt used `world_object_a` as `object_id` (19 unique groups over 43 rows) and hit a singular design matrix; fixed by grouping at the `world` level (6 groups) — logged as the one iteration this took, well within the 3-attempt budget. Final run: converged with a mild quasi-separation warning on one world-level dummy (expected at N=43, 8 parameters) but produced full output; `prg_edge_error`'s coefficient was the only one to reach p<0.05 (p=0.027) in this small, synthetic-planted sample — not a finding, the relationship was deliberately planted to test whether the fitting code could recover *something*.

Every line of output is stamped "SYNTHETIC WIRING REHEARSAL -- NOT A RESULT." Committed `3f002ce`.

## [T5] Documentation sync

- `PREREG_AMENDMENTS_PROPOSED.md`: drafted (not applied to PREREG.md itself) two amendments — (A) structural-independence checkoff citing PROJECT_STATUS §2's AST audit, (B) flagging the object-count conflict, which turned out to be a genuine **three-way** conflict on closer reading (PREREG §5 original: 5 objects w/ remote; PREREG's own 2026-07-15 amendment: 5 objects w/ stapler; INTERFACE.md §2: "Option A" = 3 objects apple/purse/glass) rather than the two-way one PROJECT_STATUS §5 item 4 described. Flagged only, not resolved, per instructions.
- `diagnosis/DATA_PROVENANCE_NOTE_PROPOSED.md`: exact proposed text for `classify_relation.py`'s docstring, `SCHEMA.md `'s §3b/§4, and `INTERFACE.md `'s §5 — all three currently cite the unrecoverable `training_data_v3_metric_only.csv` and/or a stale "SUPPORT_RELATIVE_HEIGHT_THRESHOLD is untuned" claim that was superseded by the trained Support sub-classifier in a prior session. Not applied — these are Person B's interface files.
- `PROJECT_STATUS.md`: appended a new dated "§7 Overnight session — 2026-09-20" section at the end; sections 1-6 left untouched (confirmed via reading the diff before committing).

Committed `f05aaf5`.

## Final verification (pre-handoff)

```
git diff 2a6f362..HEAD --stat -- perception-calibration-gap/PREREG.md integration/prg_config.json relation_classifier/models/ "SCHEMA.md" "INTERFACE.md"
```
Empty output — all five confirmed byte-identical to this branch's starting commit. `2a6f362` is the commit this branch was created from (see Setup section above).

---

# Session 2 — Verification pass, branch `fix/boundary-features` (2026-09-20, continued)

A follow-up session audited (not extended) the T1 finding above. Detailed,
per-task logging for this session lives in `verification/*.md` (V0-V7)
rather than duplicated here, per that session's own logging instructions.
Summary: the T1 "confirmed root cause" (interface region includes
background) was found to be **not a training-vs-live mismatch at all**
(identical code both paths) — only a small (~35-45%), separately-confirmed
ring-kernel-shape mismatch exists, and testing it against REAL
physics-verified `gazebo_contact` (obtained by successfully running
`extract_gazebo_contacts.py` for the first time ever) showed it changes
nothing: 0.500 balanced accuracy, 43/43 pairs still predicted "Separate."
No code fix was implemented (decision gate failed decisively, thresholds
not softened per instructions). See `verification/V1_AUDIT.md`,
`verification/V2_GATE.md`, `verification/RETRAIN_OR_RELABEL_PROPOSAL.md`,
`MORNING_HANDOFF_2.md`, and `PROJECT_STATUS.md` §8 for full detail.

Branch: `fix/boundary-features`, from `overnight/prg-diagnosis` @ `0a51fbb`.
Final integrity check: all 5 `.joblib` SHA256s unchanged;
`git diff unified-pipeline..HEAD --stat` on `*.joblib`, `integration/prg_config.json`,
`perception-calibration-gap/PREREG.md`, and Person-A's `conformal_gate.py`/
`gap_computer.py` all empty.

