# Morning Handoff — overnight/prg-diagnosis

Autonomous overnight session, 2026-09-20. Branch `overnight/prg-diagnosis`,
created from `unified-pipeline` @ `2a6f362`. Not merged anywhere, not
pushed. Full detail and exact commands/outputs: `OVERNIGHT_LOG.md`. Full
diagnosis reasoning: `diagnosis/BINARY_MODEL_DIAGNOSIS.md`.

## 1. Task status

| Task | Status | Evidence |
|---|---|---|
| T1a — git archaeology | Done | `training_data_v3_metric_only.csv` confirmed **unrecoverable**: never committed, at any commit, on any branch. `OVERNIGHT_LOG.md` §T1a |
| T1b — model self-consistency | Done (partial, honestly) | Model/scaler introspected fully; could NOT run predict() on real training rows (only recoverable CSV lacks 5/8 features). `OVERNIGHT_LOG.md` §T1b |
| T1c — label-noise check | Done | Reproduced 60/101 Contact disagreement exactly; found new 13/101 Separate-side disagreement. `OVERNIGHT_LOG.md` §T1c |
| T1d — real DA-V2 comparison | **Blocked**, logged | No `.pth` checkpoint anywhere in repo; `rclpy` not installed. Two independent hard blockers, not attempted further. `OVERNIGHT_LOG.md` §T1d |
| T1 — new finding beyond the brief | Done | Confirmed a code-level root cause independent of the MDE-vs-GT question: `feat_boundary_sharpness`/`feat_depth_gradient`'s interface region is background-dominated. See §2 below |
| T1e — diagnosis doc | Done | `diagnosis/BINARY_MODEL_DIAGNOSIS.md`, hypothesis-by-hypothesis verdicts |
| T2 — re-run with corrected source | Done, reframed | T1's answer wasn't "use depth source X" — it was a feature-code bug. Ran the natural equivalent: corrected-feature rehearsal, `experiments/rerun_with_corrected_interface_SIM.py` |
| T3 — wider prg_edge_error sweep | Done | 43 pairs, all 6 worlds. `diagnosis/PRG_EDGE_ERROR_SWEEP_SIM.md` + `.csv` |
| T4 — Model D / two-signal gate rehearsal | Done | `rehearsal/model_d_and_two_signal_gate_REHEARSAL.py`, runs end-to-end, fully stamped SYNTHETIC |
| T5a — PREREG amendments (proposed) | Done | `PREREG_AMENDMENTS_PROPOSED.md` — structural-independence checkoff + object-count conflict (turned out to be **3-way**, not 2-way) |
| T5b — data provenance doc note (proposed) | Done | `diagnosis/DATA_PROVENANCE_NOTE_PROPOSED.md` |
| T5c — PROJECT_STATUS.md sync | Done | New dated §7 appended, §1-6 untouched |

## 2. The T1 verdict, in three sentences, and the decision it needs from you

The binary "always Separate" behavior is now explained by a **confirmed, code-level bug**: `feat_boundary_sharpness()`/`feat_depth_gradient()` in `extract_relation_features.py` measure each object's silhouette edge against the *background* (77-92% of every frame) rather than the true object-object boundary, producing feature values 14-40 standard deviations outside the frozen model's training distribution on every tested pair — large enough, on the model's own largest-magnitude coefficient, to saturate every prediction to "Separate" regardless of anything else. This is independent of, and better-evidenced than, PROJECT_STATUS's original leading hypothesis (trained-on-MDE, tested-on-GT depth), which could not be tested tonight (no DA-V2 checkpoint file exists in the repo, `rclpy` isn't installed) and is now weakened by git history showing the relevant flag has never been anything but `gazebo_range_gt` in this repo's life. **The decision I need from you**: fixing the feature bug changes the feature *distribution* the frozen model was calibrated against, so a feature-code fix and a model retrain need to ship together, not separately — I did not retrain anything (per your rule), but you should decide whether to approve (a) the feature-code fix to `extract_relation_features.py` (Person B's file — needs her buy-in too) and (b) a retrain of `logistic_model.joblib`/`scaler.joblib` against the corrected feature definition, ideally combined with addressing the separately-confirmed label-noise issue (heuristic "Contact" disagrees with physics-verified `gazebo_contact` 60/101 times) in the same pass rather than two.

## 3. Everything here is UNVALIDATED / synthetic

Every number in `diagnosis/*_SIM.md`, `diagnosis/*_SIM.csv`, `experiments/*_SIM.py`, and `rehearsal/*_REHEARSAL.py` is ray-cast synthetic depth on staged SDF scenes (or, for T4, additionally fabricated Gap/occlusion/failure columns) — **none of it is citable as C1, C1-ext, C4, or C4-ext evidence**, and each file says so in its own header/footer per the task's labeling discipline. Nothing touched real hardware; nothing in this session ran on the physical arm, camera, or any real trial.

## 4. What Researcher A (Krupa) needs to know

- The interface-region bug is in her file (`extract_relation_features.py`, `feat_boundary_sharpness`/`feat_depth_gradient`), affects the SAME functions used by training extraction, `LiveDepthFeatureSource` (real-hardware path), and the ray-cast validator alike — it's a property of the code, not specific to any one depth source. She should see `diagnosis/BINARY_MODEL_DIAGNOSIS.md`'s H3 section and the minimal-fix suggestion before touching the file (I did NOT edit `extract_relation_features.py` — only demonstrated the fix's effect in a separate `experiments/` script, per the no-modify-production-code discipline).
- `SCHEMA.md `/`INTERFACE.md ` both still cite the untuned `SUPPORT_RELATIVE_HEIGHT_THRESHOLD` and the lost `training_data_v3_metric_only.csv` as if current — both are stale (superseded by the trained Support sub-classifier, and confirmed-unrecoverable, respectively). Exact proposed replacement text is in `diagnosis/DATA_PROVENANCE_NOTE_PROPOSED.md`; not applied, since these are her interface files.
- `prg_edge_error`'s current AUROC signal against wrong predictions (0.62, per T3) should NOT be read as validation of the two-signal design — it's most likely an artifact of the classifier's current degeneracy. Worth re-checking together once the classifier fix lands.
- If she has a local, uncommitted copy of `training_data_v3_metric_only.csv` or the world*_features.csv / world*_gazebo_contacts.csv intermediates anywhere (a laptop, an old branch, a cloud drive), it would resolve T1b's remaining gap (running the frozen model on its own real positives) far better than anything reconstructible from git.

## 5. Config / model / PREREG integrity confirmation

```
git diff 2a6f362..HEAD --stat -- perception-calibration-gap/PREREG.md integration/prg_config.json relation_classifier/models/ "SCHEMA.md" "INTERFACE.md"
```
Empty output (verified just before writing this handoff) — `integration/prg_config.json` (`use_stub: true` unchanged), every `.joblib`/scaler file, `perception-calibration-gap/PREREG.md`, `SCHEMA.md `, and `INTERFACE.md ` are all byte-identical to this branch's starting commit `2a6f362`. Nothing was retrained, overwritten, or edited in place.

## 6. Files this session added (all on `overnight/prg-diagnosis`, all committed)

- `PROJECT_STATUS.md` — new §7 appended (also committed the pre-existing untracked file itself, first commit of the session)
- `OVERNIGHT_LOG.md` — full command/output trail
- `diagnosis/BINARY_MODEL_DIAGNOSIS.md`, `diagnosis/PRG_EDGE_ERROR_SWEEP_SIM.md`, `diagnosis/prg_edge_error_sweep_SIM.csv`, `diagnosis/DATA_PROVENANCE_NOTE_PROPOSED.md`
- `diagnosis/scripts/run_binary_model_diagnostic.py`, `diagnosis/scripts/sweep_prg_edge_error_SIM.py`
- `experiments/rerun_with_corrected_interface_SIM.py`
- `rehearsal/model_d_and_two_signal_gate_REHEARSAL.py`
- `PREREG_AMENDMENTS_PROPOSED.md`
- `MORNING_HANDOFF.md` (this file)

Stop condition reached: T1-T5 all done or explicitly blocked-and-logged. Stopping here.
