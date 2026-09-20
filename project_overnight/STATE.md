# T0 — State Inventory

Branch `project/overnight-hardening`, created from `explore/geometric-prg-error`
@ `f5cac32` (verified base decision below).

## Branch decision (T0 step 1-2)

```
$ git branch -a
```
Checked, for each: `git log --oneline -3` and `git log --oneline explore/geometric-prg-error..<branch>`:

| Branch | Log -3 (head) | Commits NOT in explore/geometric-prg-error |
|---|---|---|
| unified-pipeline | 2a6f362 added classifier files | (none) |
| origin/failure-calibration | c19ecc7 Added ReadME | `c19ecc7` (1 trivial README commit) |
| origin/boundary-depth-estimation | e63cf94 added the boundary depth estimation... | (none) |
| overnight/prg-diagnosis | 0a51fbb Overnight session complete... | (none) |
| fix/boundary-features | 44a4979 V7: documentation and handoff... | (none) |
| retrain/relation-classifier | 1e72398 R9: finalize handoff docs... | (none) |
| explore/geometric-prg-error | f5cac32 E6: final handoff... | (n/a, self) |

**`explore/geometric-prg-error` is a strict linear descendant of every
other listed branch except `failure-calibration`, whose only extra commit
is a one-line "Added ReadME"** — not meaningful work. **Base = `explore/geometric-prg-error`,
no missing-commits flag needed**, confirmed by the empty diff results
above (this satisfies T0/BRANCH step 2's evidence requirement).

### Anomaly found, logged not acted on
An unrelated branch `a/overnight-hardening` exists locally (not in the
task's list), one commit ahead of `unified-pipeline` @ `2a6f362`, with its
own log at `a_overnight/LOG.md`. Its own branch-decision log shows it was
created under a DIFFERENT prompt/task variant (explicitly bases off
`unified-pipeline`, not `explore/geometric-prg-error`, reasoning that
PRG-side diagnostic work should be excluded from Person-A hardening).
This is inconsistent with THIS session's explicit instructions, which
name `explore/geometric-prg-error` as the base (with a fallback rule that
doesn't apply here per the table above). **Per hard rule ("never touch
other branches"), `a/overnight-hardening` is left exactly as found** —
noted here for a human to reconcile the two overnight-hardening attempts
if that's ever needed, not merged or deleted.

## T0 environment/package check

```
$ python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
2.13.0+cu130 True
$ python3 -c "import cv2; print(cv2.__version__, hasattr(cv2,'aruco'))"
4.9.0 True
$ python3 -c "import sklearn; print(sklearn.__version__)"           # 1.7.2
$ python3 -c "import statsmodels; print(statsmodels.__version__)"   # 0.14.6
$ python3 -c "import open3d; print(open3d.__version__)"             # 0.19.0
$ python3 -c "import ultralytics; print(ultralytics.__version__)"   # 8.4.94
$ python3 -c "import shap"
AttributeError: module 'coverage' has no attribute 'types'   # BROKEN -- numba/coverage package conflict, pre-existing env issue
$ python3 -c "import rclpy"
ModuleNotFoundError: No module named 'rclpy'                 # still missing, consistent with every prior session
$ gz sim --version                                            # Gazebo Sim 8.14.0, working
$ find / -iname "*.pth"
/home/manasi/bps-project/checkpoints/depth_anything_v2_vits.pth (+2 metric variants)  # outside repo, as before
$ python3 -c "from models.depth_anything_v2.dpt import DepthAnythingV2"
ModuleNotFoundError: No module named 'models.depth_anything_v2.dinov2_layers'  # still blocked, same as 2 prior sessions
```

**shap is newly confirmed broken** in this environment (not noted before
— T7's SHAP requirement will need a workaround or an honest UNVERIFIED
flag; see T7 below).

## T0 SHA256 baseline (protected files)

```
$ find . -iname "*.joblib" | sort | xargs sha256sum
[unchanged from all prior sessions -- see project_overnight/LOG.md for the full list]
$ sha256sum integration/prg_config.json perception-calibration-gap/PREREG.md "SCHEMA.md " "INTERFACE.md "
[recorded in LOG.md]
```

## Component inventory (EXISTS / PARTIAL / MISSING, file:line evidence)

| # | Component | Status | Evidence |
|---|---|---|---|
| T1 | Camera-to-arm extrinsic calibration | **MISSING** | `find . -iname "*calib*"` returns only SO-101 joint/URDF calibration files (`lerobot_mujoco_sim/asset/.../so101_new_calib.xml` etc.) — encoder calibration, not camera-to-robot extrinsics. No Kabsch/SVD solver, no YAML output, anywhere in the repo. |
| T1 (reusable) | ArUco marker detection + solvePnP | **EXISTS**, reusable | `perception-calibration-gap/real_hardware_adapter.py:98-154` (`detect_markers`), `:157-176` (`rvec_tvec_to_pose`) — used AS-IS for T1's marker-pose helper (do not duplicate the ArUco pipeline). |
| T2 | Grasp lookup-table execution path | **MISSING** | No `pose_source` field in `logger.py`'s `TrialRecord` (checked all 30+ fields, `perception-calibration-gap/mock-pipeline/logger.py:62-119`). No lookup-table YAML anywhere. `run_trial.py:349` (`execute_action_real`) drives a MuJoCo mock env directly from `perceived_target_xyz`, not a config-driven bin lookup — different mechanism than T2 asks for. |
| T3 | Checkpoint 1 comparison tool | **EXISTS, one real gap found** | `perception-calibration-gap/hand_calc_validation.py` already does independent hand-calc, MAE, scatter plot, reads `validation_scenes.csv`, refuses on **zero** data rows (`hand_calc_validation.py:341-343`). **Gap**: `validation_scenes.csv` currently has 3 non-empty PLACEHOLDER rows (verified: byte-identical to `validation_scenes_template.csv`, `diff` empty) whose `notes` fields literally say "REPLACE with your real measured..." — the script's own "no data rows" check does NOT catch this, so it would currently run to completion on template data and print a (meaningless) MAE/PASS-FAIL. This is the genuine PARTIAL gap to close. No test file exists for this script (`find . -iname "*test*hand_calc*"` empty) — tests + import-independence check also needed. `checkpoint1_validation_scenes.py` is a separate, earlier SCENE-DESIGN-only skeleton (its own docstring: "STATUS: design skeleton... does NOT touch validation_scenes.csv") — not a duplicate of `hand_calc_validation.py`, a different, non-conflicting stage. |
| T4 | Real camera path / live-frame ingestion | **EXISTS, largely**, needs config/replay/timing verification | `perception-calibration-gap/real_camera/capture_and_gap.py` (294 lines), `orbbec_capture.py` (135 lines), `aruco_pose.py` (87 lines) — full read pending; verify config-driven camera source, `--replay` mode, `masks_are_bbox_placeholder` at every call site, timing/readiness checks. |
| T5 | YOLO data + training pipeline | **EXISTS, substantially** | `perception-calibration-gap/yolo/train_yolo_scaffold.py:64-249` already has `try_load_pretrained_model`, `generate_synthetic_yolo_dataset` (SYNTHETIC image set, matching T5's own "tiny SYNTHETIC image set only" requirement), `try_get_real_coco8`, `run_sanity_finetune`. **Gap**: no per-object acceptance-gate evaluator with pre-committed per-object thresholds (T5's explicit "aggregate metric alone never passes" requirement) — `detector_validation.py` (below) is closest but built for a different stage. No marker-derived auto-labeling pipeline found yet (needs verification). |
| T5 (related) | Detector validation harness | **EXISTS** | `perception-calibration-gap/mock-pipeline/detector_validation.py:67-366` — per-object IoU validation, cross-object confusion, fallback-plan triggering. Different from T5's acceptance-gate evaluator (this validates a trained detector's bboxes against ground truth; T5 wants a training-time pass/fail gate) but likely reusable for its IoU/confusion logic. |
| T6 | PRG signal (sanity test, changed-interface attempt) | **PARTIAL**, prior session's C1/C2/C3 exist and failed criteria | Full history in `explore/E3_FINDINGS.md` — decisive negative result, diagnosed mechanism (bbox-ring background contamination). T6a's synthetic step-edge sanity test does NOT yet exist as an isolated unit test (only full-scene sweeps in `explore/scripts/e3_evaluate_candidates.py`) — needed to confirm/refute the diagnosed mechanism precisely. T6b's "neighbor localization from depth only" variant does not exist yet. |
| T7 | Gap-only statistics package | **EXISTS, substantially** — needs verification pass | `perception-calibration-gap/analysis/analysis_c1.py` (242 lines, 3 logistic models + ΔAUC + VIF implied by file size), `analysis_c2.py`-`analysis_c5.py`, `power_analysis_c1.py`, `power_analysis_c3.py`, `mock-pipeline/conformal_gate.py:52-168` (`ConformalGate` class: `fit`/`flag`/`empirical_coverage`, 70/30 split, alpha configurable, per-object coverage). `mock_data_generator.py:79` (`generate_mock_trials(..., gap_signal_multiplier=1.0)`) already supports a planted-vs-null effect via the multiplier arg. Full read of each analysis_c*.py file still needed to confirm SHAP, reliability diagram, KS exchangeability, FAIL-Detect baseline are present — SHAP specifically is at risk given the broken `shap` import (see environment check above). |
| T8 | Model D / two-signal gate, guarded | **NEEDS VERIFICATION** | Not yet located as a standalone module (may be inside `analysis_c4.py`, 212 lines). Hard rule 5's stub-exclusion-with-logged-count and "refuses when zero real rows" behavior not yet confirmed to exist — likely needs building regardless of what's found, since this is a hard-rule requirement with an explicit test mandate. |
| T9 | Trial harness / schema / session planning | **PARTIAL** | `logger.py`'s `TrialRecord` (`mock-pipeline/logger.py:61-119`) has `trial_id, session_id, object_id, ..., prg_edges (list[dict] with relation_prediction/relation_confidence/prg_edge_error/geometric_feature_vector/mask_is_bbox_placeholder), prg_is_stub, prg_error_note`. **Discrepancy from "Known context"**: the task describes flat scalar PRG columns (`prg_edge_error`, `prg_edge_error_is_stub`); the REAL, already-existing schema instead uses a richer per-edge LIST (`prg_edges: List[dict]`, `prg_is_stub: bool`) to support >1 neighbor per trial — this is more correct, not a bug; T9's stub-exclusion rule will be built against the REAL field names, not the ones the prompt assumed. **Missing fields** (checked against `run_trial.py`'s own grasp code, `logger.py`'s full field list): `grasp_point_xyz_camera`, `grasp_point_xyz_robot`, `pose_source`, `scene_condition`, `repeat_id`, `controller_git_hash` (a `policy_checkpoint_hash` field exists but is a different concept per its own comment — "renamed from openvla_checkpoint_hash"), `notes`, a PRG model-version field. `object_id` is a free string, not validated against `{apple, purse, glass}`. No scheduler, no `data_quality_report.py` (`find . -iname "*scheduler*" -o -iname "*data_quality*"` both empty). |
| T10 | Documentation / conflicts | **PARTIAL** | Object-count conflict already documented across two prior sessions (`PREREG_AMENDMENTS_PROPOSED.md`, `PREREG_RETRAIN_AMENDMENT_PROPOSED.md`) but not consolidated into one place; needs the Gap-only-primary / Model-D-conditional / scripted-controller-descope / lookup-table-infrastructure / PRG-history amendments this task specifically asks for. |
| T11 | Verification | Not started | Standard end-of-session pass, to be run after implementation. |
| T12 | Hardware checklist | **MISSING** (as a single ordered document) | Individual pieces exist scattered (marker manifest, `capture_and_gap.py`, `REAL_CAPTURE_PROTOCOL_V2.md`) but no single ordered, time-estimated, exit-criteria checklist exists. |
| T13 | Handoff | Not started | End-of-session deliverable. |

## Conflicts with "Known context" found so far

1. **PRG schema field names**: the prompt's "Known context" and T9's
   instructions assume flat `prg_edge_error`/`prg_edge_error_is_stub`
   columns; the actual, already-committed schema (`logger.py`) uses
   `prg_edges: List[dict]` (one dict per neighbor, containing
   `prg_edge_error` inside each) and `prg_is_stub: bool`. Built against
   the real schema throughout this session, not the assumed one.
2. **`validation_scenes.csv` state**: the prompt says "still a blank
   template" — technically true (byte-identical to the template file)
   but the template itself contains 3 non-empty placeholder rows with
   explicit "REPLACE with..." notes, not zero rows. The existing
   `hand_calc_validation.py` only refuses on literally zero rows, so it
   does NOT currently refuse on this actual blank-template state — a
   real gap, closed in T3 below.
3. **T3's two-script situation**: `checkpoint1_validation_scenes.py`
   (scene DESIGN, explicitly self-described as a skeleton) and
   `hand_calc_validation.py` (the actual COMPARISON tool T3 asks for)
   are two different, non-duplicate, complementary files — worth stating
   explicitly so neither gets mistaken for the other or rebuilt
   unnecessarily.

Full per-task work log: `project_overnight/LOG.md` (appended after every
task, per instructions).
