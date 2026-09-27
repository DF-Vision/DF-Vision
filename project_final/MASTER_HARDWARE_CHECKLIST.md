# Master Hardware Checklist — Both People, One Ordered Document

Merges `project_overnight/HARDWARE_CHECKLIST.md` (Manasi's side:
calibration, arm, lookup table, Checkpoint 1, YOLO capture) and
`explore/REAL_CAPTURE_PROTOCOL_V2.md` (Krupa's side: camera-only
real-depth PRG capture with measured ground truth) into one ordered list,
with explicit parallelization notes. Nothing in this list has been done
by any autonomous session — every step needs the physical arm, camera,
markers, or objects. No step here was executed or simulated by this
consolidation pass.

## Dependency structure (read this first)

```
Step 1 (print + measure markers) ─── shared prerequisite for everything below
        │
        ├── Track A (Manasi, needs the ARM): 2 → 3 → 4 → 5 → 9 (sequential)
        │
        ├── Track B (Manasi, camera + markers only, NO arm needed):
        │       6 (Checkpoint 1) — can run BEFORE or DURING Track A,
        │       does not need camera-to-arm extrinsic calibration (step 3)
        │
        ├── Track C (Krupa, camera + markers only, ARM OFF):
        │       7 (real-depth PRG capture) — fully independent of Track A,
        │       can run any time after step 1, in parallel with everything else
        │
        └── Track D (either person, camera + markers, benefits from but
                doesn't strictly require Track A's calibration):
                8 (YOLO image capture) — raw capture can start any time after
                step 1; auto-labeling quality improves once step 3's extrinsic
                exists, but is not blocked on it
```
**The single most parallelizable fact**: Track C (Krupa's real-depth PRG
capture, step 7) needs ONLY the camera and step 1's markers — it can
start immediately after step 1 and run entirely independently of
Manasi's arm work (Tracks A/steps 2-5). There is no reason to wait for
arm calibration before doing it.

## 1. Print and measure markers (~20 min) — shared prerequisite

Print `perception-calibration-gap/generated_markers/*.png` at **100% /
Actual Size** (disable "fit to page" — a scaled print silently corrupts
every downstream occlusion number). Ruler-measure each against
`manifest.csv`: apple 3.5cm, glass 4.5cm, purse 5.5cm.
**Exit criterion**: every printed marker's measured edge matches
`manifest.csv`'s `size_cm` to within 1mm, or reprint at the correct scale.

---
### Track A (Manasi, arm required, sequential)

## 2. LeRobot servo calibration + teleop check (~30-45 min)
Vendor's own SO-101/LeRobot calibration procedure (outside this repo's
scope). **Exit criterion**: smooth, repeatable teleop through full range,
no calibration warnings.

## 3. Camera-to-arm extrinsic calibration (T1, ~20-30 min)
```bash
python3 perception-calibration-gap/calibration/capture_marker_points.py \
    --image <captured_frame.png> \
    --intrinsics perception-calibration-gap/calibration/camera_intrinsics.yaml \
    --marker-sizes-cm 0:5.0,1:5.0,2:5.0
# hand-measure each marker's ROBOT-BASE-FRAME position, fill into point_pairs.csv
python3 perception-calibration-gap/calibration/compute_extrinsic_calibration.py \
    --points perception-calibration-gap/calibration/point_pairs.csv \
    --output perception-calibration-gap/calibration/T_base_from_camera_optical.yaml
```
**Exit criterion**: held-out RMS < 2cm, YAML saved (if the script
refuses — degenerate points, unit mismatch, high residual — follow its
error message and re-capture; do not override the refusal).

## 4. Fill the grasp lookup table (~30-60 min, iterative)
Teleoperate to a good grasp pose per table position, record joint angles,
add a `bins` entry (~9-12 bins, 3x3-ish grid) to
`perception-calibration-gap/grasp/lookup_table_config.yaml`.
**Exit criterion**: every bin within `joint_limits`; a dry-run test point
lands on the expected bin (`lookup_table_executor.py`, dry-run only,
never `--i-am-supervising` without a human at the terminal watching).

## 5. Three-consecutive-reliable-grasps gate (~15-30 min)
Under YOUR direct supervision, 3 different bins in a row.
**Exit criterion**: all 3 succeed (picked up, no collision, no
joint-limit fault) before trusting the table for real trials.

---
### Track B (Manasi, camera + markers only — can run before/during Track A)

## 6. Checkpoint 1: ground-truth marker validation scenes (~45-90 min)
**Does NOT need step 3's arm-to-camera extrinsic** — this validates
`occlusion_estimator.py`'s hand-calc-vs-pipeline agreement using camera +
markers + measured object geometry only, independent of the robot base
frame. Stage 5-8 known-geometry scenes on the apple + 2-3 spot-checks per
other object; fill real measurements into
`perception-calibration-gap/validation_scenes.csv` (replacing the
placeholder rows — `hand_calc_validation.py` REFUSES to run on the
template, by design, confirmed working: `tests/test_hand_calc_validation.py`,
mutation-tested `project/closure` C7).
```bash
python3 perception-calibration-gap/hand_calc_validation.py
```
**Exit criterion**: MAE < 5 percentage points (script prints PASS/FAIL).

---
### Track C (Krupa, camera only, ARM OFF — fully parallel to Tracks A/B)

## 7. Real-depth PRG signal capture (~60-90 min) — see `explore/REAL_CAPTURE_PROTOCOL_V2.md` in full
**This is the ONLY real-depth test any PRG candidate has ever had** — see
`project_final/PRG_SIGNAL_STATUS.md` for why this matters: every prior
result (5 sessions, 3 design approaches, all decisive negatives) was on
simulated/ray-cast depth only.

Stage 14 scenes (apple/purse/glass + a flat prop, Contact/Support/Separate
combinations spanning true gap 0-150mm, including 2mm/5mm near-contact
gaps via a calipers-measured spacer method) per the protocol's own table.
Measure true gaps with calipers/ruler BEFORE capturing frames — write
numbers down immediately. 5 RGB+depth frame pairs per scene.
```bash
python3 explore/scripts/e5_analyze_real_capture_gaps.py \
    --capture-dir real_capture_v2_YYYYMMDD/ \
    --bbox-csv real_capture_v2_YYYYMMDD/bbox.csv
```
**Exit criterion**: not pass/fail — this IS the actual sim-to-real
validity test. Report AUROC/error numbers honestly, compare against
`explore/E3_FINDINGS.md`'s simulation numbers (C1=0.457→0.280 post-fix,
C2=0.486→0.390, C3=0.463→0.286 — see `PRG_SIGNAL_STATUS.md` items #4/#6)
**and, if useful, also against the C5 neighbor-aware candidate**
(`explore/scripts/geometric_prg_error_neighbor_aware.py`, built
`project/closure` — postdates this protocol, so
`e5_analyze_real_capture_gaps.py` does NOT evaluate it out of the box;
verified this session by reading the script, which imports
`geometric_prg_error` (C1/C2/C3) directly with no pluggable candidate
mechanism — including C5 would need a small script edit first, not
something this consolidation session did since it would be new code, not
a merge/verify). Note plainly that every result here still has
`mask_is_bbox_placeholder=True` (no real segmenter exists).

---
### Track D (either person, benefits from Track A but not blocked on it)

## 8. YOLO image capture (~90-120 min, spread across sessions)
Per object (apple, purse, glass), real images across varied poses,
lighting, backgrounds, occlusion, distances — 50-100+ images/object.
Auto-labeling infrastructure exists (`perception-calibration-gap/yolo/auto_label_from_markers.py`,
built + tested `project/closure` C3, 15/15 tests) but has only ever been
run on synthetic marker images — running it on real captured images for
the first time is this step's actual remaining work.
**Exit criterion**: per-object image count logged; revisit
`per_object_gate.py`'s placeholder thresholds (currently precision/
recall/AP50 > 0.5, chosen to exercise gate logic on synthetic data, not
as a considered real-detector bar) once real training is attempted.

---
### Final step (needs Tracks A AND B both passed)

## 9. First live smoke trials (~30-60 min)
Once steps 1-6 all pass: run `perception-calibration-gap/run_trial.py`
against real hardware (not the MuJoCo mock), arm under direct
supervision, a handful of trials.
```bash
python3 perception-calibration-gap/analysis/data_quality_report.py \
    --log perception-calibration-gap/mock-pipeline/data/<your_real_log>.jsonl
```
**Exit criterion**: no session/object confounding flagged, no
controller-freeze violation, `pose_source` populated correctly on every
trial.
