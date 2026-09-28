# DF-Vision — Complete Software & Simulation Summary

**Generated**: 2026-09-28, from `project/closure-actions` (the branch that
consolidates every prior session's work — `project/closure`,
`project/audit`, `project/prereg-correction`, `project/final-integration`,
`retrain/relation-classifier`, `project/overnight-hardening`,
`overnight/prg-diagnosis`, `explore/geometric-prg-error`,
`fix/boundary-features`, plus this session's own closure tasks).

Purpose of this document: one place that lists, in full and with reasons,
everything that is **done** in software/simulation, and everything that
is **still remaining** and why. It does not repeat every prior status
document's internal history — those are linked throughout — it
synthesizes them into a single readable account.

---

## 1. The big picture, in one paragraph

The project builds a perception-failure-prediction and safe-deferral
system for a SO-101 robot arm doing pick-and-place under camera
occlusion: a "Calibration Gap" signal (the divergence between what a
camera perceives and what's geometrically true), a conformal statistical
gate that decides when to defer to a human, and (attempted, ultimately
retired) a second signal from a Physical Relation Graph (PRG) of
object-to-object contact relationships. Every piece of **software** for
this pipeline is built, tested, and validated on **synthetic and
simulated data** — Gazebo worlds, ray-cast renderers, fabricated trial
generators. **Zero real hardware trials have ever been run.** That is the
single fact that determines almost everything in the "remaining work"
section below: it is not more code that's missing, it's real data.

---

## 2. What is DONE — software

### 2.1 Core perception pipeline

| Component | File(s) | Status |
|---|---|---|
| Camera-to-arm extrinsic calibration solver | `perception-calibration-gap/calibration/compute_extrinsic_calibration.py`, `capture_marker_points.py` | Built, tested. Refuses on degenerate points/unit mismatch/high residual rather than silently producing a bad calibration. |
| ArUco marker generation + geometry manifest | `perception-calibration-gap/generate_markers.py`, `generated_markers/`, `manifest.csv` | Built. Markers generated and sized; **not yet printed/measured** (that's real-world step 1, see §4). |
| Occlusion estimation (ground-truth geometric) | `perception-calibration-gap/gazebo/occlusion_estimator.py` | Built, self-tested (partial-occlusion case etc.), used as the "true visibility" side of every Gap computation. |
| Masked TSDF fusion (perceived visibility) | `perception-calibration-gap/gazebo/tsdf_fusion.py` | Built, self-tested on synthetic depth frames with a synthetic occluder. |
| Calibration Gap computation | `perception-calibration-gap/gazebo/gap_computer.py` | Built. Computes `Gap_initial`, `Gap_max`, `Gap_slope` from perceived vs. true visibility. |
| Grasp lookup table + dry-run executor | `perception-calibration-gap/grasp/lookup_table_config.yaml`, `lookup_table_executor.py` | Infrastructure built; dry-run mode enforced (refuses `--i-am-supervising` without a human watching). **Table itself is empty** — needs real teleop data (real-world step 4). |
| Real-camera capture + Gap pipeline | `perception-calibration-gap/real_camera/capture_and_gap.py` | Built, includes a `--replay` mode against a synthetic fixture (`generate_replay_fixture.py`) that reproduces 4/4 good + 4/4 correctly-flagged cases. Never run against a live camera feed for a real trial. |
| Trial schema + logger | `perception-calibration-gap/mock-pipeline/logger.py` (`TrialRecord`, `TrialLogger`) | Built, schema-complete (every field used by every downstream analysis script, including all fields added across sessions T9/C2). |
| End-to-end pipeline wiring rehearsal | `perception-calibration-gap/daily_integration_check.py`, `e2e_integration_rehearsal_SYNTHETIC.py` | Both run and pass — proves every real module plus the (stub) PRG and (stub) explain() components are wired and schema-compatible, on a fabricated synthetic scene. Explicitly never cited as a scientific result — a wiring check only. |

### 2.2 Object detection (YOLO)

| Component | File(s) | Status |
|---|---|---|
| Auto-labeling from ArUco markers | `perception-calibration-gap/yolo/auto_label_from_markers.py` | Built, tested (15/15 tests, `project/closure` C3). Only ever run on **synthetic** marker images — never on a real photograph. |
| Per-object detection gate | `perception-calibration-gap/yolo/per_object_gate.py` | Built. Placeholder thresholds (precision/recall/AP50 > 0.5) chosen to exercise the gate's logic on synthetic data, explicitly **not** a considered bar for a real detector. |
| YOLO training scaffold | `perception-calibration-gap/yolo/train_yolo_scaffold.py` | Built. Never trained on a real-image dataset (none exists yet — that's real-world step 8). |

### 2.3 Statistics / conformal gate (C1-C5, PREREG.md's own numbering)

| Contribution | What's built and validated | Why it's not "finished" |
|---|---|---|
| **C1** — Gap as an independent failure predictor | `analysis/analysis_c1.py`: full model fit, SHAP explainability (with a guarded fallback for a broken local `shap` install — permutation + tree-native importance — independently verified against a real, working SHAP install in an isolated venv), per-object breakdown table. Runs correctly end-to-end on synthetic mock data. | Whether Gap predicts REAL failure needs real trials; separately, the study is underpowered at the currently-assumed effect size (§3). |
| **C2** — Post-hoc recalibration | `analysis/analysis_c2.py`: linear and MLP correction paths, both run on synthetic data. | Reports neither correction beats the uncorrected baseline — but this is a synthetic-only negative result; the real question needs real miscalibration to correct against. |
| **C3** — Gap × occlusion × clutter interaction | `analysis/analysis_c3.py`, plus `PREREG.md`'s own pre-registered power justification (lines 84-90) already reclassifies this as exploratory (power 0.20-0.30 at the planned N), not confirmatory, **before any data collection**. | Same as above — synthetic-only, and pre-emptively downgraded on honest power grounds, not a code gap. |
| **C4** — Conformal failure gate | `mock-pipeline/conformal_gate.py`, `analysis/analysis_c4.py`: FAIL-Detect-style GMM baseline, identical conformal calibration procedure applied to both Gap and the baseline, 3-signal ablation, KS exchangeability checks across sessions and objects, split-discipline enforcement (a previously-vacuous assertion was fixed and is now regression-tested, `tests/test_split_discipline.py`). All run correctly on synthetic data. | Real coverage/efficiency needs real trials. |
| **C5** — Active mitigation via TSDF re-fusion | `analysis/analysis_c5.py` | Synthetic-only wiring check; recovery rate reported honestly as 0/2 per its own pre-committed criterion — not evidence the real mechanism doesn't work, just that the fabricated synthetic scenario didn't demonstrate it. |
| **Model D** — combined Gap+PRG model | `analysis/model_d.py`, 11/11 tests passing | Fully built, but by design refuses to run unless explicitly enabled AND fed real (non-stub) PRG data — neither is true, and per the PRG retirement decision (§3), this path is now formally out of scope for the paper. |

### 2.4 PRG (Physical Relation Graph) work — simulation, all attempts complete and conclusively negative

This is the largest single body of work in the project and deserves its
own detailed accounting, because "done" here means **decisively
falsified in simulation**, not "abandoned incomplete." Full chronological
detail lives in `project_final/PRG_SIGNAL_STATUS.md`; summarized here:

1. **Legacy trained binary "touching vs. not" classifier** — predicted
   "Separate" on effectively 100% of tested pairs. Root-caused (a
   ring-kernel train/serve mismatch, later shown to be a minor
   contributing factor, not the main cause) and confirmed genuinely
   broken against real physics-verified ground truth
   (`gazebo_contact`): balanced accuracy exactly 0.500, 43/43
   "Separate" predictions before and after the fix.
2. **Full retrain on 330 new pairs with real, physics-verified
   `gazebo_contact` labels**, leave-one-world-out cross-validation with
   a genuinely held-out final world. Best of 6 candidate models: 0.667
   mean CV balanced accuracy (gate: >0.75), with **every** candidate
   showing near-zero Contact-class recall (0.00-0.27) — a structural
   limitation of the depth-derived 8-feature set, not a bad training
   run.
3. **Oracle-ceiling check**: ground-truth geometric features (true 3D
   surface gap from SDF geometry, zero camera/depth involvement)
   recovered much better Contact recall (0.649 vs. 0.00-0.27),
   indicating the depth-derived features specifically are lossy — not
   that Contact/Support are physically indistinguishable.
4. **Three classifier-free geometric candidates (C1/C2/C3)**: 0 of 5
   non-held-out Gazebo worlds cleared the pre-committed sensitivity bar
   for any candidate; validity AUROC at or below chance. Root cause
   diagnosed precisely: an omnidirectional ring sample around a
   target's bounding box has no notion of *which direction* a real
   neighbor lies, so it's dominated by whatever's nearest in any
   direction — frequently background clutter, not the true touching
   partner.
5. **A separate 0.0-return bug** in C1's gap estimator was found and
   fixed (the sampling ring was straddling the object's own bounding
   box). Fixing it did **not** fix problem #4 — re-evaluation showed
   the AUROC got measurably *worse* (0.457 → 0.280), independently
   confirmed via a proper paired, world-grouped statistical bootstrap
   (95% CI on the difference excludes zero) rather than a naive
   overlapping-interval eyeball check.
6. **A fourth, differently-designed candidate** (depth-only,
   connected-components-based neighbor identification, built
   specifically to fix mechanism #4) does work correctly on idealized
   synthetic test scenes that defeat the earlier candidates — but still
   fails decisively (0/5 worlds) against the same pre-committed
   criteria on the real evaluation worlds, with a different failure
   signature (consistently negative correlation, rather than mixed-sign
   near-zero).

**Net result, formally recorded**: 3 fundamentally distinct design
approaches, 5 dedicated sessions, all measured against the same
pre-registered, never-loosened criteria (`explore/CRITERIA.md`),
**all failed**. `integration/prg_config.json`'s `use_stub: true` has
never been changed through any of this. This is genuinely complete,
rigorous, negative science — not unfinished work.

**Decision (2026-09-27, Manasi/Krupa), recorded and closed out**: the
dual-signal PRG extension (C1-ext, C4-ext, Model D) is formally
**retired** for the paper submission. Gap-only (C1, C4) is the sole
primary path, which `PREREG.md`'s own pre-registered cut strategy already
treats as fully valid on its own. The one real-depth capture test that
has never been run (Track C, see §4) is still worth doing once real data
exists, but purely as confirmatory/appendix material, not as a blocking
dependency, and not as grounds to attempt a fifth PRG design.

### 2.5 Statistics infrastructure and correctness

- **Power analysis correction**: a specific bug (`0d74cd4`) in
  `mock_data_generator.py` coupled the effect-size knob to Gap's own
  noise/variance, producing a stale, invalid "0.79-0.80 power at N=195"
  figure that had been sitting uncorrected in `PREREG.md`. Root-caused
  precisely, fixed, and the corrected power table (no cell reaches the
  conventional 0.80 target — max is 0.60 at 3x effect, N=195) is
  reproducible on demand (`project_prereg_correction/scripts/full_power_table.py`).
  **This correction has now been formally applied to `PREREG.md` itself**
  (`## Amendment 2026-09-27 — Power analysis correction`, a pure append,
  byte-verified to touch zero existing text).
- **Mutation testing**: 30/30 kill rate across 10 named safety-critical
  guards (calibration solver, lookup-table/out-of-reach checks, dry-run
  enforcement, marker-size config validation, template-CSV refusal,
  Model D stub-exclusion, session scheduler, data-quality report,
  controller-freeze check, depth-unit guard) — re-verified with no
  regression at every consolidation checkpoint.
- **Paired, world-grouped bootstrap statistics**: used to rigorously
  confirm (not just eyeball) that a measured AUROC change is real, with
  a proper accounting for the world-clustering structure of the data
  (`project_audit/C4_AUDIT.md`).
- **Split-discipline enforcement**: a previously *vacuous* assertion
  (`assert ... or True`, which could never fail) guarding against
  training-set/test-set leakage in the FAIL-Detect baseline was found and
  fixed with real enforcement by trial ID, now regression-tested
  (`tests/test_split_discipline.py`).
- **Structural independence audit**: an AST-based static check
  (`explore/scripts/check_structural_independence.py`) confirming no
  circular or hidden coupling between independently-developed modules —
  built, extended, and still passing at every checkpoint.

### 2.6 Gazebo / simulation environment

- **6 Gazebo worlds**, used as the evaluation substrate for every PRG
  candidate and for the DA-V2 depth pipeline. A malformed, redundant
  world-level `gz-sim-label-system` plugin declaration in all 6 `.sdf`
  files (causing a load error) was found and fixed (commented out, not
  deleted), confirmed via a bounded headless `gz sim` smoke test.
- **`LiveDepthFeatureSource`'s 8 geometric features** were validated
  against an independently, from-scratch-built analytic ray-caster with
  closed-form expected-value formulas (zero shared code with the
  function under test) across all 6 worlds and 25 real object pairs:
  **182 pass / 0 fail / 18 documented, non-arbitrary skips** (curved
  surfaces, camera-viewpoint-dependent edge cases).
  Script: `relation_classifier/scripts/validate_live_depth_feature_source.py`.
  This is the most rigorous single validation exercise in the project —
  an independently-derived ground truth, not a self-referential check.
  `gz sim` itself is confirmed installed and working headless (Gazebo
  Sim 8.14.0).
- **DA-V2 (Depth Anything V2) → point cloud → grasp-centroid pipeline**
  was built and validated across all 6 Gazebo worlds (Krupa's side) —
  this is the depth backbone actually intended to feed real/live camera
  frames, as opposed to Gazebo's own geometrically-exact simulated depth.
  Note: the DA-V2 checkpoint itself lives outside this repo and `rclpy`
  is not installed in this environment, so no live DA-V2 inference has
  been run **inside this repo** — see §4/#4 in Remaining Work.
- **A relation classifier retrained specifically against real,
  physics-verified `gazebo_contact` ground truth** (not a heuristic
  label) — 330 pairs, strict train/serve feature parity, leave-one-
  world-out CV. This IS real supervision from the simulator's physics
  engine, which is why its clean negative result (near-zero Contact
  recall) was taken seriously as a structural finding rather than
  dismissed as "just needs more data."
- **A 3-feature logistic-regression Support/Contact sub-classifier**
  (trained on 176 usable labeled rows) replaced a single hand-picked
  threshold that topped out at 0.529 balanced accuracy (barely above
  chance, confirmed via proper group-aware 5-fold CV). The new model
  reaches F1=0.659/balanced accuracy=0.706, statistically distinguishable
  from the old threshold (McNemar's p<0.0001).

### 2.7 Test suite, mutation testing, and code health

- **154 tests passing, 1 skipped, 0 failing** (`tests/` directory),
  covering every module above. The single skip is environment-specific
  (a broken local `shap` install due to a `coverage`/`numba` version
  conflict — the fallback path IS tested and passing, and the real SHAP
  path was independently verified working in an isolated venv).
- **Fresh-clone reproduction** confirmed the entire suite plus the
  end-to-end synthetic rehearsal, the replay-mode smoke test, and a
  key statistics function all reproduce identically from a clean git
  clone at the consolidation commit.
- **A harness pluggability improvement**: `e5_analyze_real_capture_gaps.py`
  (the script that will run Track C's real-depth evaluation once data
  exists) now supports a `--candidate` flag (`c1`/`c2`/`c3`/
  `c5-neighbor-aware`/`all`), proven byte-identical to the pre-change
  script for the legacy candidates via a golden-output regression test,
  so the harness is ready to evaluate any candidate (including the
  newer C5 design) the moment real data exists — without having
  generated or claimed any performance number for it yet.

---

## 3. What is DONE — documentation and process integrity

These aren't simulations, but they are core to "is this project
scientifically sound," so they belong in a done/remaining accounting:

- **`PREREG.md`** (the frozen pre-registration document) has been
  correctly amended, append-only, twice: once (2026-07-14/15, prior
  session) and once (2026-09-27, this session, the power-analysis
  correction + PRG retirement note). Both amendments were verified
  byte-for-byte to add content without altering a single existing line —
  a discipline enforced and checked, not just claimed.
- **A full resolution log** (`project_final/RESOLUTION_LOG.md`) tracks
  every "needs a human decision" flag raised across 9+ sessions: of 14
  distinct issues, 8 are fully resolved with passing tests or applied
  decisions, 4 are drafted-but-unapplied documentation edits waiting on
  a human to paste them into frozen files by hand (by design — no
  autonomous session may edit `PREREG.md`/`SCHEMA.md`/`INTERFACE.md`
  except via pre-drafted, reviewed amendment text), and 2 are genuinely
  open decisions requiring human judgment (see §4).
- **Branch hygiene**: every session's branch ancestry was explicitly
  checked (`git merge-base --is-ancestor`) rather than assumed; the only
  two non-ancestor commits found across the whole repo history were
  individually inspected and correctly rejected (one superseded, one
  containing false claims — see §4/README below).

---

## 4. What is REMAINING — and why

Everything below requires physical action, a human decision, or an
external resource — **none of it is missing code**.

### 4.1 Real hardware trials (the big one)

**Nothing here has been done or simulated.** Every statistical result
(C1-C5) is on synthetic/fabricated data. The full ordered checklist is
`project_final/MASTER_HARDWARE_CHECKLIST.md`; summarized:

| Step | What | Blocked on |
|---|---|---|
| 1. Print + measure ArUco markers | Physical printing, ruler measurement | A printer and a ruler — trivial but physical |
| 2. LeRobot servo calibration + teleop | Vendor procedure | The physical SO-101 arm |
| 3. Camera-to-arm extrinsic calibration | Hand-measured robot-base-frame marker positions | Steps 1-2, and physical camera placement |
| 4. Fill the grasp lookup table | Teleoperated grasp poses per table bin | Step 2 (arm), iterative human teleop |
| 5. Three-consecutive-reliable-grasps gate | Supervised live grasping | Steps 2-4 |
| 6. Checkpoint 1 (ground-truth marker validation) | Staged real scenes, hand measurement | Camera + markers only — **no arm needed, can run any time** |
| 7. **Real-depth PRG capture (Track C)** | 14 staged scenes, caliper-measured true gaps, RGB+depth capture | Camera + markers only — **fully independent of the arm**, the single highest-value remaining item since it's the only real-depth test any PRG design has ever had |
| 8. YOLO real-image capture | 50-100+ real photos per object, varied conditions | Camera only, no arm |
| 9. First live smoke trials | Real `run_trial.py` runs under supervision | ALL of steps 1-6 passing first |

**Why this is genuinely the critical path, not busywork**: the project's
own `PROJECT_STATUS_FINAL.md` labels every C1-C5 result
"BLOCKED-ON-HARDWARE" specifically because the *code* is done and
synthetic-validated — what's missing is real signal to run it on. No
further simulation, refactor, or algorithm design changes this. The
single most parallelizable, highest-leverage step is **#7 (Track C)**:
it needs only a camera and printed markers (no arm), can start
immediately, and is the one thing that could ever move the PRG
conclusion (§2.4) — though per the 2026-09-27 decision, even a positive
Track C result would only be appendix/confirmatory material, not a
reason to reopen PRG as a primary contribution.

### 4.2 The true effect size for power analysis

`POWER_STATUS.md` is explicit about this: the entire power table (§2.5)
is built on `TRUE_COEFS["b_gap"] = 2.5`, an arbitrary, hand-specified
constant with no citation to a pilot dataset or literature value anywhere
in the repo. **No amount of further simulation can resolve this** — it's
a fact about the real world that only exists once real trials are run.
The moment real pilot data exists, re-running the (already-built,
already-correct) `power_analysis_c1.py` with the actual observed effect
size replaces every number in the table with a real one. This is a
one-line data-substitution away from being solved, but the data doesn't
exist yet.

### 4.3 Two genuinely open human decisions (not code gaps)

1. **Object-count conflict**: `PREREG.md` §5 says "5 physical objects"
   (with a documented stapler substitution amendment); `INTERFACE.md` §2
   Option A says "3 physical objects." This has been flagged by 4+ prior
   sessions and deliberately left untouched by every one of them,
   including this one — it needs Manasi's decision on which object set
   is actually current, not a technical resolution. A draft amendment
   already exists (`PREREG_AMENDMENTS_PROPOSED.md` items 3, 8) ready to
   paste in once decided.
2. **DA-V2 checkpoint provisioning**: where the DA-V2 model weights
   should live (committed to the repo, symlinked, or documented as an
   external dependency) is an infrastructure/logistics decision, not a
   technical one. The checkpoint currently exists outside the repo
   (`/home/manasi/bps-project/checkpoints/`), `rclpy` is not installed in
   this environment, and no DA-V2 inference path exists inside this repo
   as a result — this blocks nothing in software (the pipeline that
   would consume it is already built and tested against synthetic
   stand-ins) but does block ever running DA-V2 inference for real here.

### 4.4 Drafted-but-unapplied documentation amendments

Four items are fully written, reviewed, and zero-risk, but sit unapplied
because this project's own discipline requires a human (not an
autonomous session) to paste amendment text into the three frozen
documents (`PREREG.md`, `SCHEMA.md`, `INTERFACE.md`):

- The structural-independence checklist item (evidence already exists
  and passes: `explore/scripts/check_structural_independence.py`).
- Several more `PREREG.md` amendments (Gap-only-as-primary framing,
  Model-D-conditional-on-PRG framing, lookup-table infrastructure notes,
  the consolidated PRG history) — all still accurate as of this session,
  just waiting on sign-off and manual paste-in.
- `SCHEMA.md`'s proposed field additions documenting fields the code
  already has (`relation_model_version`, `stage_timings_ms`,
  `prg_edge_error_mode`, and 9 more from an earlier session) — a
  pure documentation-sync gap; the code itself is already complete.
- A data-provenance note for `training_data_v3_metric_only.csv`, cited
  in `SCHEMA.md`/`INTERFACE.md`/`classify_relation.py` but confirmed
  unrecoverable from git history (never committed, on any branch) — the
  fix is a documentation note explaining this, already drafted.

None of these block any code from running; they are purely "the frozen
docs haven't caught up to what the code already does."

### 4.5 A README correction, prepared but not yet merged

`origin/failure-calibration`'s top-level `README.md` claims "✅ PRG
Integration," "✅ Hardware Pipeline," and "✅ Voice Interaction" as
complete — all three are false per the findings above (PRG: decisively
retired negative result; hardware: zero real trials ever; voice:
never started). A corrected version has been written and committed on a
separate local branch, `fix/readme-status-correction` (pushed to
`origin/fix/readme-status-correction`, **not** merged into
`failure-calibration` — that's a shared branch, and merging it needs
human review first, not an autonomous decision).

### 4.6 Two small, explicitly out-of-scope items

- **Voice interaction**: never started at all — no code, no design, not
  part of any session's scope so far. Listed in `PROJECT_STATUS.md`'s
  original roadmap but not a contribution this project's statistical
  claims (C1-C5) depend on.
- **`test_real_adapter.py`'s own frame-synthesis helper has a bug**
  (a Y-axis convention mismatch in `synthesize_marker_frame()` that
  silently mirrors every synthetic test frame it generates, root-caused
  precisely via direct instrumentation). This is a bug in a **test
  script**, not in the real `real_hardware_adapter.py` code under test
  — that code is already verified correct via a different, working
  synthesis path (`generate_replay_fixture.py`, 14/14 tests passing).
  Fixing the buggy test script is legitimate future work but isn't
  blocking anything, since the code it would test is already covered.

---

## 5. One-paragraph honest bottom line

Every piece of software this project needs — the Gap/occlusion/TSDF
pipeline, the conformal gate, the statistics and power-analysis
correction, YOLO auto-labeling infrastructure, and three fully-explored,
decisively-negative PRG signal designs across six validated Gazebo
worlds — is built, tested (154 passing tests, 30/30 mutation kills,
fresh-clone-reproducible), and honestly documented. **What remains is not
more code or more simulation — it is physical hardware use** (printing
markers, calibrating the arm, capturing real RGB-D frames, running real
trials) plus a small number of decisions only Manasi and Krupa can make
(the object-count conflict, DA-V2 checkpoint provisioning, and signing
off on already-drafted documentation amendments). No further autonomous
software session should attempt new PRG designs, retrain any model, or
otherwise treat this as an open engineering problem until real capture or
trial data exists to work with.
