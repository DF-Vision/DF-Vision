# Project Status — Calibration Gap × PRG Dual-Signal Study

**Generated:** 2026-09-20, after a multi-session engineering pass over the PRG/relation-classifier side of the codebase. This file reconciles the original planning documents (Merged Project Plan, BPS Pipeline Status, Master Reference paper plan, Master Integration Plan — all supplied by the user) against the actual, hands-on-verified state of the code as of today.

**Ownership** (per the original plan, confirmed by git commit authorship): **Manasi — Person A**, Hardware & Statistics owner (SO-101 arm, camera rig, marker calibration, YOLO, TSDF/Gap, all stats, conformal gate math). **Krupa — Person B**, Perception & Multimodal owner (boundary/relation feature extraction, PRG construction, relation classifier, and — per the plan — the NLG/TTS/ASR layer, which does not appear to have been started in this codebase at all).

**How to read this file**: every claim below is tagged with how it was verified. `[VERIFIED THIS SESSION]` means directly run/tested with real output inspected. `[FROM PLANNING DOCS]` means asserted in the supplied planning documents but not independently re-checked in this pass. `[INFERRED]` means pieced together from code/git evidence, not directly stated anywhere. Nothing here is guessed without saying so.

---

## 1. The single most important finding: the PRG relation classifier's first-stage model looks broken on anything but sim-ground-truth-style depth — unresolved, root cause not yet confirmed

This is the finding most likely to derail the dual-signal claim (MC1 / C1-ext) if not resolved first, so it leads this document rather than being buried in a table.

**What was found** `[VERIFIED THIS SESSION]`: the pre-trained binary "touching vs. not" model (`relation_classifier/models/logistic_model.joblib` + `scaler.joblib` — trained by Krupa's `train_relation_classifier.py`, not touched this session) classifies **every tested pair as "Separate" at ~100% confidence**, including pairs staged and geometrically confirmed as genuine Contact and genuine Support, when fed synthetic depth from an analytically-exact ray-cast renderer (built and validated this session — see §3). Real, meaningfully different feature values go in (boundary_sharpness, depth_gradient, occlusion_boundary_score all shift substantially between scenes); the same "Separate, ~1.0 confidence" answer comes out regardless.

**What was ruled out** `[VERIFIED THIS SESSION]`: this is not a missing-noise artifact. Added a literature-grounded depth-noise model (edge blending matching the SDF's own declared `<anti_aliasing>4</anti_aliasing>`, Nguyen et al. 2012 distance-scaled Gaussian noise, 1mm quantization) and re-ran — no change. Stress-tested at 5x and 15x that noise magnitude — still no change, confidence unmoved to six decimal places on the Contact case. This rules out "the synthetic depth just isn't noisy enough" as the explanation.

**Leading hypothesis, not yet confirmed** `[INFERRED]`: `extract_relation_features.py` has a `PIPELINE_VALIDATION_MODE` flag that switches the *training-data-generation* depth source between `gazebo_range_gt` (geometrically-exact simulator depth — what this session's ray-cast renderer approximates) and `mde_predicted_z` (Depth Anything V2's learned, relative/metric depth estimate — what the BPS Pipeline Status document says is the *only* depth backbone actually wired to real/live frames, since BP-MDE was permanently descoped). If the deployed binary model was trained on MDE-derived features rather than GT-geometric ones, its whole decision boundary lives in a different part of feature space than anything a geometrically-exact renderer (noisy or not) can reach — which would fully explain why noise injection didn't help: the mismatch isn't noise, it's depth *source*. This has not been confirmed — nobody has checked which depth source actually produced the 189-row training set the model was fit on (the file it names, `training_data_v3_metric_only.csv`, no longer exists under that name anywhere in the repo).

**A second, independent contributing factor already on record** `[VERIFIED EARLIER THIS SESSION]`: the model's own training labels are noisy — `heuristic_label` (the AABB static-pose heuristic) disagrees with `gazebo_contact` (the physics-verified signal) on 60 of its own 101 "Contact" calls. A model trained partly against a heuristic that's wrong more often than not on its own positive class would plausibly learn an overly conservative, "when in doubt say Separate" decision boundary independent of any depth-source issue.

**Why this matters for the paper, precisely**: MC1's entire premise is that PRG_edge_error is a *second, meaningful* signal alongside Gap. If the relation classifier upstream of it defaults to "Separate" on real or realistic input, `classify_relation()` never reaches the new Support/Contact sub-classifier (validated separately and working correctly — see §3), and `compute_prg_edge_error()` downstream gets fed a `relation` value that may not reflect reality. This is not yet a proven dead end — the MDE-vs-GT hypothesis is testable and, if confirmed, has a straightforward-sounding fix (test with the depth source the model actually saw) — but it is unresolved, and no one should assume PRG_edge_error is trustworthy on real hardware until it's chased down.

**Concrete next diagnostic step, not yet done**: run real DA-V2 inference (already implemented and validated per BPS Status §P1, working across all 6 Gazebo worlds) on one of the same staged Track-A ray-cast scenes' RGB, and feed *that* depth through `classify_relation()` instead of the geometrically-exact ray-cast depth. If the binary model behaves sensibly on DA-V2 depth specifically, the MDE-vs-GT hypothesis is confirmed and the fix is "always test/deploy with DA-V2 depth, not GT-style synthetic depth." If it's still broken, the problem is the model itself (likely the label-noise issue), which is a materially bigger fix (retraining against better ground truth).

---

## 2. Status by component (BPS Pipeline Status structure, P1–P4)

| Component | Owner | Plan's original status | Actual current status | Evidence |
|---|---|---|---|---|
| P1 — Six-world DA-V2 depth → point cloud → grasp-centroid pipeline | B (Krupa) | COMPLETE, validated across all 6 Gazebo worlds | **Unchanged / believed complete** | `[FROM PLANNING DOCS]` — not independently re-verified this session; no reason to doubt it, but not re-run either |
| P2 — Grasp-point computation, camera→world | B (Krupa) | Camera-optical-frame centroid only, no arm-frame transform | **Unchanged** | `[FROM PLANNING DOCS]` |
| P3 — Live relation classifier / PRG feeding real frames | B (Krupa) | Per BPS status doc: **did not exist**, only `.sdf`-ground-truth features | **Now exists and is largely validated** — this is the single biggest change since the planning docs were written | `[VERIFIED THIS SESSION]` — see §3 below in full |
| P4 — Real-hardware bring-up (Orbbec Gemini 2, SO-101) | A (Manasi) | Camera detected; arm not calibrated; no IK | **Unchanged — still the hard gate on any real trial** | `[FROM PLANNING DOCS]`, consistent with PREREG.md's own "NOT STARTED" markers (see §4) |
| Structural independence (PRG side ↔ detector/TSDF/Gap side) | Both | Pending as of PREREG.md's 2026-07-15 amendment: Person B's side confirmed by inspection; Person A's side of the claim explicitly marked **NOT YET CONFIRMED** | **Now confirmed, both directions, rigorously** | `[VERIFIED THIS SESSION]` — AST-parsed every import in `extract_relation_features.py`, `classify_relation.py`, `compute_prg_edge_error.py`, `build_prg.py`, plus every Person-A file (YOLO scaffold, detector validation, TSDF fusion, Gap computer, occlusion estimator, grasp estimator) and did a transitive BFS closure on both. Zero internal imports at all in any of these 10 files — not even within their own side. Zero overlap. **PREREG.md's pending checklist item can now be checked off with a citation to this audit** — it has not been, since PREREG.md hasn't been touched (see §5, remaining work) |

---

## 3. What actually got built and validated this session (the PRG side, in detail)

This is new work layered on top of what Krupa had already committed (`extract_relation_features.py`, `classify_relation.py`, `compute_prg_edge_error.py`, `build_prg.py`, `integration/prg_client.py` — all pre-existing, authored by her per git history). Nothing below was assumed working without being run.

| Item | Status | Evidence |
|---|---|---|
| `LiveDepthFeatureSource`'s 8 geometric features, validated against independently hand-derived ground truth | **Done** | Built a from-scratch analytic ray-caster + closed-form expected-value formulas per feature (no shared code with the function under test), ran across all 6 worlds, 25 real object pairs. **182 pass / 0 fail / 18 documented, non-arbitrary skips** (curved-surface/mixed-face views for `surface_normal_consistency`; camera-viewpoint-dependent non-adjacency for `edge_continuity`). Script committed: `relation_classifier/scripts/validate_live_depth_feature_source.py` |
| `worlds/*.sdf` Gazebo Label-system config bug | **Fixed** | All 6 world files had a malformed, redundant world-level `gz-sim-label-system` plugin declaration causing a load error; commented out (not deleted) in all 6, confirmed via a bounded headless `gz sim` smoke test that the error disappears and nothing else breaks |
| `perception-calibration-gap/real_camera/capture_and_gap.py` — `masks_are_bbox_placeholder` flag | **Fixed** | Was never passed on the real-hardware smoke-test call site despite using dilated-bbox placeholder masks — meant real smoke-test edges would have been silently mislabeled as real-segmentation-derived. Now passed correctly |
| `SUPPORT_RELATIVE_HEIGHT_THRESHOLD` (a single hand-picked cutoff on one feature) → real trained sub-classifier | **Replaced, integrated, validated** | The old threshold, at every value tested, topped out at **balanced accuracy 0.529 — barely above the 0.500 chance floor** (confirmed via a proper group-aware 5-fold CV sweep, not just eyeballed). Replaced with a 3-feature Logistic Regression (C=1.0), trained on all 176 usable labeled rows, reaching **F1=0.659 / balanced accuracy=0.706** in the same CV discipline. A Random Forest was also evaluated (F1=0.730/bal.acc=0.759, both higher) but LogReg was chosen for production for **stability** (fold-to-fold std 0.06-0.09 vs RF's 0.13-0.14) and interpretability, since McNemar's test showed the two aren't statistically distinguishable from each other (p=0.50) but both are statistically distinguishable from the old threshold (p<0.0001, overwhelmingly in the new models' favor). New artifacts: `relation_classifier/models/support_logistic_model.joblib`, `support_scaler.joblib`, `support_summary.txt`, `relation_classifier/scripts/train_support_classifier.py` |
| `integration/prg_client.py`'s three code paths (stub / real-success / exception-fallback) | **All three independently verified** | Real success path: confirmed `is_stub=False`, real (non-None) relation/confidence data, exact 6-field SCHEMA §3b shape, deterministic (recomputed identically twice). Empty-list path (`prg_edges == []`, the specific thing flagged as "moved and at risk" by a refactor): confirmed `is_stub=False` (not `True`, not `None`) on a genuinely isolated object with zero adjacent pairs — no bug found, contrary to the concern. All three verified via a monkeypatch of `_load_config()` in-memory only — `integration/prg_config.json`'s committed `use_stub: true` was never touched (confirmed via `git diff` after every check, not just inferred) |
| Full-chain smoke test: `LiveDepthFeatureSource → classify_relation() → build_prg()` on staged Contact/Support/Separate scenes | **Run — surfaced §1's finding** | This is exactly where the binary-model OOD issue above was discovered. `build_prg()` itself is correct (produces the right graph structure for whatever relation it's handed); the problem is entirely upstream, at the binary touching-model stage |

---

## 4. Hard blockers — cannot be closed by any amount of AI compute, need a human and/or physical hardware

Per PREREG.md's own explicit amendments (dated 2026-07-14/15) and the BPS status doc, these were already known blockers before this session; nothing here has changed:

- **Checkpoint 1 (ground-truth marker validation): NOT STARTED.** `perception-calibration-gap/validation_scenes.csv` is still byte-identical to the blank template — re-confirmed this session with a direct `diff`. This is a **hard gate**: nothing downstream (real Gap computation, real trial collection) is trustworthy without it, per the plan's own rules.
- **Fiducial markers: generated as PNGs, never physically printed or ruler-measured.** Per PREREG.md's amendment: "must be completed before any marker is mounted on the rig... marker-size misconfiguration silently corrupts every downstream occlusion number."
- **Camera-to-arm extrinsic calibration: not done.** Explicit identity placeholder, intentionally left unset rather than guessed (per both PREREG.md and BPS status doc).
- **Inverse kinematics for the SO-101, or the lookup-table fallback: not decided, not built**, per BPS status and the Master Integration Plan's Day-1 P0 item.
- **No real trial data of any kind exists.** Every trial-shaped output produced anywhere in this session (`daily_integration_check.py`, `run_trial.py`'s stub-path output, the full-chain smoke test) is explicitly synthetic/wiring-validation only, never cited as evidence for any statistical claim — consistent with how the code itself labels these outputs.
- **YOLO fine-tuning**: needs self-captured real images per the plan; none have been captured (would require the physical camera + physical objects + a human to label/capture).

None of the above can be meaningfully advanced by an unattended AI session tonight. Putting compute time against them would just produce more synthetic dry-runs of already-validated wiring, not real progress.

---

## 5. What an unattended AI session genuinely CAN finish tonight (software-only, no hardware needed)

Ordered by dependency — each depends on the ones above it:

1. **Chase the §1 finding to a root cause.** Run real DA-V2 depth inference (already built, per BPS status P1) on the same staged Track-A scenes and re-test the binary touching model against that depth instead of GT-style ray-cast depth. This is the single highest-value thing to do first — it either resolves the biggest open risk to MC1/C1-ext, or definitively rules out the leading hypothesis and points at the training-label-noise explanation instead, which is a different (bigger) fix.
2. **If the MDE-hypothesis is confirmed**: document it precisely (which depth source is safe to test/deploy against) and re-run the full-chain smoke test (§3, last row) using DA-V2 depth, to confirm Contact/Support/Separate now vary sensibly.
3. **If it's not confirmed** (model is genuinely miscalibrated regardless of depth source): this becomes a retraining task — revisit whether `heuristic_label` is trustworthy enough as a training target at all, or whether `gazebo_contact` (physics-verified) should be weighted more heavily; this needs a scoping decision, not silent action, given how consequential retraining the frozen binary model would be.
4. **Update PREREG.md with the overdue amendments** (its own stated discipline — dated, appended, never edited in place): (a) the structural-independence checklist item can now be checked off with a citation to this session's audit; (b) the object-vocabulary/scripted-controller/3-object descope described in the "Master Reference" document isn't reflected in PREREG.md's Section 5 (which still lists 5 objects with a remote→stapler substitution) — worth a dated amendment reconciling which version is actually current, since two of the supplied planning documents currently disagree with each other on object count (5 vs. 3) and this file shouldn't quietly pick one without flagging the conflict to you.
5. **Wire and dry-run Model D / the two-signal conformal gate against synthetic data only**, explicitly labeled as a pipeline-wiring rehearsal (matching the discipline `daily_integration_check.py` already uses) — proves the statistical code path runs end-to-end with real (non-stub) PRG_edge_error values feeding in, without pretending it's a real result.
6. **Extend Track A-style validation to `compute_prg_edge_error()`'s output values themselves** on a wider sweep of staged scenes now that the classifier feeding it is understood — currently only spot-checked on a handful of scenes.
7. **Documentation sync**: `SCHEMA.md`/`INTERFACE.md` still cite a `training_data_v3_metric_only.csv` file that doesn't exist under that name anywhere in the repo — worth tracking down or noting as lost provenance, since it's cited as the source of specific statistics quoted in `classify_relation.py`'s own docstring.

## 6. What NOT to let an overnight run drift into

- Do not let it attempt anything resembling real trial data collection, hardware calibration, or IK — it will produce nothing trustworthy and will burn the whole 7 hours on something no software fix can close.
- Do not let it retrain the binary touching model without an explicit decision from you first (per this session's consistent discipline: report findings, don't quietly fix consequential things). If step 1-3 above point toward retraining, that should come back as a proposal, not a fait accompli by morning.
- Do not let it touch `integration/prg_config.json`'s `use_stub` value in the committed file — every verification this session used in-memory overrides specifically to avoid this, and that discipline should hold.
- Do not treat any synthetic/dry-run output (this session's or a new overnight one) as citable evidence for C1/C1-ext/C4/C4-ext — every existing dry-run script already says this about itself; a new one should too.

---

## 7. Overnight session — 2026-09-20 (autonomous, branch `overnight/prg-diagnosis`)

Full detail in `diagnosis/BINARY_MODEL_DIAGNOSIS.md`, `diagnosis/PRG_EDGE_ERROR_SWEEP_SIM.md`,
`OVERNIGHT_LOG.md`, and `MORNING_HANDOFF.md`. This section only summarizes;
those files are the source of truth.

**§1's finding, updated**: the leading hypothesis this document previously
named (model trained on `mde_predicted_z`, tested on `gazebo_range_gt`-style
depth) could not be tested tonight — no DA-V2 checkpoint file exists in this
repo and the `rclpy` runtime `depth_publisher.py` needs isn't installed here;
logged as a blocker rather than guessed at. It is also weakened by new
evidence: `extract_relation_features.py` has exactly one commit in its
entire git history, and `PIPELINE_VALIDATION_MODE` has been `True`
(`gazebo_range_gt`) in every version of that file that has ever existed —
there is no git trace of the training CSV ever being generated in
`mde_predicted_z` mode.

A different, code-provable root cause WAS confirmed instead: `feat_boundary_sharpness()`
and `feat_depth_gradient()` (both in `extract_relation_features.py`) define
their "interface" region as effectively the full silhouette ring of BOTH
objects, not the shared object-object boundary — so on every tested pair,
these two features are dominated by each object's edge against the
BACKGROUND, not against each other. On this session's ray-cast depth this
produces `boundary_sharpness` z-scores of 14.5–43.6 (against the frozen
model's own training-time scaler stats) in every single case, and
`boundary_sharpness` carries the model's largest-magnitude coefficient —
enough alone to saturate `predict_proba` toward "Separate" regardless of
every other feature. A synthetic rehearsal correcting only this region
(not the frozen model, not any committed file) restored real variation in
the model's predictions (0/15 → 7/15 non-Separate on the same 15 staged
pairs) and roughly doubled binary accuracy (27% → 60%), though it did not
fully fix Contact-class predictions specifically — consistent with this
document's already-on-record label-noise finding (heuristic_label
disagrees with gazebo_contact on 60/101 of its own Contact calls) being a
second, compounding, independent issue.

**Practical upshot**: no depth source (real DA-V2 or otherwise) can be
expected to fix the frozen model's current behavior on its own, because
the primary confirmed defect is in feature computation, not depth source.
A real fix needs both a `feat_boundary_sharpness`/`feat_depth_gradient`
code change (Person B's file) AND a retrain of the frozen model against
the corrected feature definition (the two must ship together, since the
frozen scaler/model were fit on the old, buggy feature distribution) —
proposed, not applied, in `diagnosis/BINARY_MODEL_DIAGNOSIS.md`'s
recommendation section, pending your decision per this project's
retrain-needs-a-human-decision rule.

**Also newly confirmed** (git archaeology): `training_data_v3_metric_only.csv`
— the file the frozen binary model was actually trained on, and the file
`SCHEMA.md `/`INTERFACE.md `/`classify_relation.py` cite statistics
against — was **never once committed to this repository**, on any branch,
at any commit. The `.gitignore` added in the same commit as the model
files explicitly excludes every OTHER generated intermediate CSV in that
pipeline (`heuristic_vs_gazebo_comparison*.csv`, `flagged_objects.csv`,
`*_gazebo_contacts.csv`, etc.); the v3 file fell through the same
"generated files aren't committed" gap. This is stated plainly as
unrecoverable rather than guessed at further. Proposed (not applied) doc
fixes for the resulting stale citations are in
`diagnosis/DATA_PROVENANCE_NOTE_PROPOSED.md`.

**Other overnight findings**: a wider `compute_prg_edge_error()` sweep (43
adjacent pairs across all 6 staged worlds, up from a handful of spot
checks) found AUROC=0.62 between `prg_edge_error` and wrong-relation
predictions — a weak positive signal, but per this session's own
interpretation, likely an artifact of the classifier's current near-total
degeneracy rather than evidence the two-signal design is working (see
`diagnosis/PRG_EDGE_ERROR_SWEEP_SIM.md`). A synthetic wiring rehearsal
(`rehearsal/model_d_and_two_signal_gate_REHEARSAL.py`) confirmed Model D
and a max-combination two-signal conformal gate run end-to-end against
real (non-stub) `prg_edge_error` values — code-path-only, not a
statistical result. A three-way object-count conflict across PREREG.md
(twice, disagreeing with itself) and INTERFACE.md was found and flagged
(not resolved) in `PREREG_AMENDMENTS_PROPOSED.md`.

**Untouched, confirmed**: `integration/prg_config.json`, every `.joblib`
model/scaler file, and `perception-calibration-gap/PREREG.md` are
byte-identical to this branch's starting commit (`2a6f362`) — verified via
`git diff 2a6f362..HEAD --stat` on those exact paths, empty output.

---

## 8. Verification session — 2026-09-20 (autonomous, branch `fix/boundary-features`), auditing §7 above

A follow-up session was tasked specifically with auditing (not assuming)
§7's finding, per the standard "every claim traces to logged output, being
right matters more than finishing." Full detail in `verification/V1_AUDIT.md`,
`verification/V2_GATE.md`, `verification/RETRAIN_OR_RELABEL_PROPOSAL.md`.
This section summarizes; those files are the source of truth.

**§7's central claim is downgraded, not confirmed.** The audit found that
`feat_boundary_sharpness`/`feat_depth_gradient`'s "interface region
includes background" property, which §7 called "the confirmed root
cause," is **not a training-vs-live code mismatch at all** — both paths
call the exact same function with the exact same interface-region logic
(`ring_a | ring_b`). The ONE confirmed, code-quoted train/live mismatch
for these two features is much narrower and smaller: the RING KERNEL
SHAPE (`scripts/export_masks.py`'s training-time boundary exporter uses a
5x5 ellipse; `LiveDepthFeatureSource`'s `_ring_from_mask()` uses a 3x3
square), worth roughly a 35-45% effect on `boundary_sharpness`'s
magnitude — nowhere near enough to explain the 14-40 standard-deviation
outliers §7 measured. `depth_gradient` has zero confirmed mismatch at all.

**The decisive new evidence**: `relation_classifier/scripts/extract_gazebo_contacts.py`
— 471 lines, its own docstring explicitly says it had **never been run**
against a live Gazebo process — was executed for real this session, for
the first time, successfully, against all 6 staged worlds (after a first
attempt pointed at the wrong SDF directory and was corrected), producing
real physics-verified `gazebo_contact` labels for 652 object pairs with
zero unresolved results. Re-running §7's rehearsal against this REAL
ground truth, using ONLY the confirmed ring-kernel fix (not §7's
"true mutual boundary" definition, which was never what the training
pipeline actually computed — an invented improvement, not a recovered
one), produced a decisive negative result: **balanced accuracy 0.500,
predicting "Separate" on all 43/43 tested pairs, identically in both the
old and "fixed" code paths.** §7's reported "fix restores 60% accuracy"
result is now understood to have worked only because it used a
definition training never used, not because it recovered anything real.

**Also corrected**: §7's "MDE-vs-GT hypothesis weakened" is walked back to
plain INCONCLUSIVE — a previously-unread code comment in
`extract_relation_features.py` (lines 170-177) states the intended NORMAL
pipeline mode is `mde_predicted_z`/`False`, with `gazebo_range_gt`/`True`
described as a one-time validation-only mode — evidence that, if anything,
cuts the other way from §7's lean. And §7's "no DA-V2 checkpoint exists
anywhere" is corrected: a checkpoint exists on this machine, just outside
this repository (`/home/manasi/bps-project/checkpoints/depth_anything_v2_metric_hypersim_vits_NewModified.pth`) —
`rclpy` is still missing, but a direct (non-ROS) inference path may not
need it.

**Decision gate result: STOP.** Per the task's pre-committed thresholds
(balanced accuracy >0.70, CI lower bound >0.55, more than one class
predicted), which failed outright rather than marginally, no code fix was
implemented. `verification/RETRAIN_OR_RELABEL_PROPOSAL.md` proposes three
ordered next steps instead: (A) regenerate training labels from the
now-demonstrated-working real `gazebo_contact` harness, (B) capture one
real Gazebo depth-camera render and check whether the background-domination
property that's confirmed in ray-cast depth also appears there (this
determines whether `feat_boundary_sharpness`/`feat_depth_gradient` need a
genuine redesign before any retrain is worth attempting), (C) retrain only
after A and B, evaluated in `experiments/` first. None of this has been
executed — it is a proposal awaiting your decision.

**Untouched, re-confirmed**: all 5 `.joblib`/scaler files identical by
SHA256 to this session's own start; `integration/prg_config.json` and
`PREREG.md` byte-identical to both `unified-pipeline` and this session's
branch point (`overnight/prg-diagnosis` @ `0a51fbb`) — see
`verification/V6_NONREGRESSION.md`.

---

## 9. Retrain session — 2026-09-20 (autonomous, branch `retrain/relation-classifier`)

Full detail in `retrain/GATES_RESULTS.md`, `retrain/models/r3_training_summary.txt`,
`MORNING_HANDOFF_3.md`. This section summarizes.

Given §8's conclusion (no confirmed, sufficient code-level fix for the
binary model's degeneracy), this session attempted an actual, disciplined
retrain: pre-committed, un-softened acceptance gates written BEFORE any
training run (`retrain/GATES.md`); 330 newly-generated, fully-provenanced
labeled pairs using REAL physics-verified `gazebo_contact` (not the
heuristic label) plus SDF-geometry-derived Support/Contact splitting, 6
worlds x 6 camera viewpoints, randomized documented depth noise per scene
(`retrain/data/relation_dataset_v2_SIM.csv` + `MANIFEST.md`, solving the
lost-training-data-provenance problem for good); strict train/serve
parity (every feature comes from the same, real `LiveDepthFeatureSource.compute()`
production code, no separate training-time feature path); and
leave-one-world-out CV with a fully held-out final test world
(`world6_dense_mixed`), never touched during model selection.

**Result: the pre-committed gates fail, decisively, not marginally.**
Best of 6 candidate models tried (2 architectures x 3 algorithms):
0.6668 mean CV balanced accuracy (gate: >0.75) — and, more importantly,
**every one of the 6 candidates showed near-zero recall (0.00-0.27) on
the "Contact" class specifically**, while Separate and Support were both
learnable to a reasonable degree. This is not one bad model or one bad
run; it is consistent across architecture and algorithm, and it persists
with clean labels and correct, parity-enforced features — pointing at a
structural limitation of the current 8-feature set for distinguishing
side-by-side Contact from stacked Support or non-touching, at least on
the depth this session could generate.

**Decision: `relation_model_version` (a new, additive config key in a new
file, `relation_classifier/relation_model_config.json` — `integration/prg_config.json`
was never touched) stays at its committed default, `"legacy"`.** No
promotion is recommended. Per this project's own stated discipline, this
is reported as a failure report and a proposal, not softened or spun as
a partial success.

**What this session did produce of lasting value, independent of the
negative training result**: (a) the dataset-provenance fix described
above; (b) a working, tested `relation_model_version` switch
(`classify_relation()`'s public return contract is unchanged and
regression-tested, `tests/test_relation_model_v2_regression.py`, 7/7
passing — the repo's first real pytest suite); (c) confirmation, for the
first time with real evidence rather than inference, that the ceiling on
this feature set is real; (d) `retrain/REAL_CAPTURE_PROTOCOL.md`, a
ready-to-run protocol for the actual sim-to-real test, which only Manasi
can execute (needs the physical camera).

**Untouched, re-confirmed**: all 6 legacy `.joblib`/scaler/CSV files
identical by SHA256 to this session's start (`retrain/BASELINE.md` vs
`retrain/R7_VERIFICATION.md`); `integration/prg_config.json`, `PREREG.md`,
and Person-A's `conformal_gate.py`/`gap_computer.py` all confirmed
untouched via `git diff`.

---

## 10. Exploration session — 2026-09-20 (autonomous, branch `explore/geometric-prg-error`)

Full detail: `explore/E1_FINDINGS.md`, `explore/E3_FINDINGS.md`,
`MORNING_HANDOFF_4.md`. Two threads, both concluded with pre-committed,
un-softened negative or mixed results.

**Thread 1 (E1) — is the §9 Contact-recall ceiling a label problem or a
feature problem?** Computed ground-truth geometric "oracle" features
(true 3D surface gap, vertical offset, horizontal overlap, contact-normal
alignment) directly from SDF geometry — no camera, no depth, no
`LiveDepthFeatureSource` involvement at all — and re-ran the same
leave-one-world-out CV as §9's retrain. **A plain logistic regression on
oracle features recovered 0.649 CV Contact recall, versus 0.00-0.27 for
every depth-derived candidate in §9.** This shifts the balance of
evidence toward "the depth-derived features are lossy," not "Contact and
Support are physically indistinguishable classes" — true-gap
distributions for Contact vs Separate are cleanly separated (medians
1.2mm vs 35.5mm), arguing against a label-tolerance explanation. The
result is not clean, though: the oracle model's Contact recall dropped to
0/4 on the held-out world, and overall balanced accuracy (0.826 CV) fell
just short of the pre-committed 0.85 "succeeds" bar — reported as a
genuine middle ground, not rounded up.

**Thread 2 (E2/E3) — can `prg_edge_error` be computed without the broken
classifier at all?** Implemented three classifier-free geometric
candidates (gap-uncertainty, boundary-sharpness-ambiguity, and a fixed
combination) behind a new, off-by-default config switch
(`relation_classifier/prg_edge_error_config.json`, separate from
`integration/prg_config.json`). **Decisive negative result against
pre-committed gates**: none of the three candidates showed any noise/blur
sensitivity (0 of 5 worlds cleared the sensitivity bar for any
candidate), and validity AUROC was at or below chance (0.457-0.486) for
distinguishing accurate from inaccurate gap estimates. Diagnosed the
mechanism directly: sampling a ring around a target's own bounding box
(the only interface the classifier-free design permits) inherits the
same background-contamination pathology found in two prior sessions'
audits of `feat_boundary_sharpness` — without knowing which direction a
specific neighbor lies, the ring is dominated by background, not the
touching partner. Per the module's own pre-committed discipline (formula
fixed before running, not adjusted after seeing results), this was not
patched in this session — logged as a specific, fixable-in-principle
proposal for later.

**No promotion of anything**: `relation_model_version` stays "legacy",
`prg_edge_error_mode` stays "classifier" (its committed default).
`integration/prg_config.json`, all `.joblib`/CSV artifacts, `PREREG.md`,
and Person-A's files confirmed untouched (`explore/E4_VERIFICATION.md`).

**What's durable from this session regardless of the negative results**:
a from-scratch, camera-independent geometric oracle sampler
(`explore/scripts/oracle_geometry.py`) usable for future feature-design
work; a precisely diagnosed, generalizable failure mode (bbox-ring
background contamination) that now explains a recurring pattern across
three separate feature-design attempts in this project's history; and
`explore/REAL_CAPTURE_PROTOCOL_V2.md`, a ready-to-run protocol for
measuring real gaps by hand, which is the only way left to test whether
any of this generalizes beyond simulation.

---

## 11. Overnight hardening session — 2026-09-21 (autonomous, branch `project/overnight-hardening`)

Full detail: `project_overnight/STATE.md`, `project_overnight/CRITERIA.md`,
`project_overnight/LOG.md`, `project_overnight/MORNING_HANDOFF.md`. This
session covered the whole system (hardware bring-up software, trial
harness, statistics, PRG signal) rather than one component, per an
explicit completeness-policy inventory (EXISTS/PARTIAL/MISSING with
file:line evidence for every T1-T9 component, `project_overnight/STATE.md`).

**Built fully (were MISSING)**: T1 camera-to-arm extrinsic calibration
(Kabsch/SVD, held-out validation, degeneracy/units/residual refusal,
8/8 tests) reusing the existing ArUco/solvePnP code rather than
duplicating it; T2 grasp lookup-table execution path (no IK solver, no
motion, dry-run only, 10/10 tests); T8 Model D (guarded off by default,
hard-rule stub-exclusion with loud logged counts, refuses on zero real
rows, 11/11 tests); T9's `data_quality_report.py` and
`session_scheduler.py` (10/10 tests).

**Hardened EXISTING code (found substantial, did not rebuild)**: T3's
`hand_calc_validation.py` (added a template-refusal guard that was
genuinely missing — the real `validation_scenes.csv` would have silently
passed through as if measured, now correctly refused; 11 new tests, AST
import-independence + 6 analytic-geometry cases); T5's per-object YOLO
acceptance gate (the aggregate-only-never-sufficient rule, 6 tests) added
alongside the already-solid `train_yolo_scaffold.py`; T7's Gap-only
statistics package verified against a synthetic-recovery check that
found and fixed a real bug in `mock_data_generator.py` (`gap_signal_multiplier=0.0`
was degenerately zeroing Gap's own variance, not just its effect —
100/100 singular-matrix failures on null data before the fix, 5/100
false-positive rate — essentially exact nominal calibration — after);
T9's `TrialRecord` extended additively with 9 fields the paper needs
(`grasp_point_xyz_camera/robot`, `pose_source`, `scene_condition`,
`repeat_id`, `controller_git_hash`, `task_success`, `notes`,
`prg_model_version`) — this surfaced and required fixing a real
regression in `daily_integration_check.py`'s field-completeness check.

**T6 (PRG signal, one bounded attempt)**: the synthetic step-edge sanity
test found the shared depth-noise model's default quantization silently
masks blur sensitivity at the Contact/Support gap scale (~1mm), and found
a NEW, more fundamental bug in C1's gap estimator (returns exactly 0.0
even on an idealized, single-edge, no-background synthetic scene — a
ring-percentile design flaw distinct from the background-contamination
mechanism found in the prior session). Documented, not fixed (xfail-
marked test) — the changed-interface variant (T6b) and full
re-evaluation (T6c) were explicitly deferred given the session's overall
scope, not silently dropped. `use_stub` remains `true`; no flip proposed.

**Flagged, not built this session**: T4's real-camera `--replay` mode and
config-driven source selection (genuinely missing, lower priority than
finishing the session's required deliverables); `rf_shap_check()`
(broken `shap` install in this environment, worked around via an
in-process module stub for everything else, this one function remains
genuinely UNVERIFIED).

**Untouched, confirmed**: `integration/prg_config.json` (`use_stub: true`),
every legacy/v2 `.joblib`, `PREREG.md`, `SCHEMA.md `, `INTERFACE.md ` all
byte-identical to session start (`project_overnight/LOG.md`'s T0 SHA256
baseline, re-checked in `project_overnight/MORNING_HANDOFF.md`). No arm
or motor code was executed; every grasp-facing script defaults to
dry-run and `--i-am-supervising` was never passed.

---

## 12. Closure session — 2026-09-21 (autonomous, branch `project/closure`)

Full detail: `project_closure/LEDGER.md`, `project_closure/CRITERIA.md`,
`project_closure/LOG.md`, `project_closure/MORNING_HANDOFF.md`. This
session's purpose was CLOSURE — finish everything §11's session deferred
or flagged, and prove the existing guards actually work (mutation
testing), rather than covering new ground.

**Closed (were flagged/deferred in §11)**: T4's `--replay` mode
(`capture_and_gap.py`, config-driven camera source, a documented depth-
unit conversion+guard, synthetic fixtures with 4 good + 4 deliberately
faulted trials, 14 tests) — also found and fixed that `capture_and_gap.py`
had never actually been importable in this environment before this
session (a broken `sys.path`, not just untested). `rf_shap_check()`
(guarded `shap` import so `analysis_c1.py` is importable regardless,
a permutation+tree-native-importance fallback with an explicit
`importance_method` marker, verified against real SHAP in an isolated
project-local venv — found a genuine N=90 sample-size sensitivity in
pooled SHAP ranking along the way). `train_yolo_scaffold.py`'s
mechanics, run for real this time (2-epoch training, all three loss
components confirmed decreasing, checkpoints save/reload/predict) —
plus a genuinely missing auto-labeling pipeline, a class-index single
source of truth, a session-grouped train/val split, and detection-metrics
known-answer tests, none of which existed before this session.

**T6's C1 gap estimator, fixed and re-evaluated**: root cause traced
(the ring straddled the bbox border, ~40-50% trivially same-surface by
construction) and fixed (exterior-only ring). Re-running
`explore/CRITERIA.md` unchanged shows the fix does NOT change, and by
one metric worsens, the E3 negative result on real multi-object scenes —
the idealized-edge bug and the real-scene background-contamination
mechanism are distinct root causes.

**Mutation-tested 10 guards** (calibration solver, lookup-table/out-of-
reach, dry-run/`--i-am-supervising`, marker-size config, template-CSV
guard, Model D stub-exclusion, session scheduler, `data_quality_report`,
controller-freeze check, depth-unit guard): 30 mutations, 30/30 killed
after closing one real test gap (joint-limit boundary inclusivity) found
along the way — plus a genuine harness bug (stale bytecode cache could
have produced false "killed" results) found and fixed mid-task.

**Re-established the synthetic statistics claims at the task's mandated
scale** (1000 null / 500×3 planted / 100+ repeated conformal splits, up
from the prior session's 50-100): the null-generator fix holds up at
20x the repeat count; a genuinely important new finding — C1's planned
study is severely underpowered (0.08-0.11 power at N=90-180, still only
0.587 at N=1440) at the mock generator's own assumed effect size,
verified not to be an implementation bug. Also found and fixed a
literally-vacuous `assert ... or True` split-discipline check in
`analysis_c4.py`.

**Built an end-to-end integration rehearsal** (SYNTHETIC) chaining
replay → Gap+PRG-stub → calibration (real solver, synthetic known
transform) → lookup-table dry-run → logger → data-quality (confounding +
stub-counts + a controller-freeze check proven to actually fire) →
Gap-only stats → conformal gate — using only this session's already-
verified code, added to the test suite so it stays green.

**Untouched, confirmed**: `integration/prg_config.json` (`use_stub: true`),
every legacy/v2 `.joblib`, `perception-calibration-gap/PREREG.md`,
`SCHEMA.md `, `INTERFACE.md ` all byte-identical to session start (SHA256
baseline in `project_closure/LOG.md`'s C0 entry, re-checked in C10). No
arm or motor code was executed; every grasp-facing test asserts the
`MotorInterface` mock is called zero times in dry-run and
`--i-am-supervising` was never passed.
