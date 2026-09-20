# Hardware Checklist — Physical Tasks Only Manasi Can Do

Ordered. Each step names its exact tooling (built or verified this
session) and its exit criterion. Nothing in this list is done by any
autonomous session — every step needs the physical arm, camera, markers,
or objects.

## 1. Print and measure markers (~20 min)
Print `perception-calibration-gap/generated_markers/*.png` at **100% /
Actual Size** (disable "fit to page" in your printer dialog — a scaled
print silently corrupts every downstream occlusion number).
Ruler-measure each against `manifest.csv`:
- apple: 3.5cm
- glass: 4.5cm
- purse: 5.5cm
**Exit criterion**: every printed marker's measured edge length matches
`manifest.csv`'s `size_cm` to within 1mm. If not, reprint at the correct
scale before mounting anything.

## 2. LeRobot servo calibration + teleop check (~30-45 min)
Follow the SO-101/LeRobot calibration procedure (outside this repo's
scope — the vendor's own calibration script). Confirm you can teleoperate
the arm smoothly through its full range without hitting joint limits or
erratic jumps.
**Exit criterion**: teleop is smooth and repeatable; no calibration
warnings from the LeRobot tooling.

## 3. Camera-to-arm extrinsic calibration (T1, ~20-30 min)
```bash
# 1. Fill perception-calibration-gap/calibration/camera_intrinsics.yaml
#    (copy from camera_intrinsics_template.yaml) with your REAL Orbbec
#    Gemini 2 intrinsic calibration.
# 2. For each of >=6 marker positions spanning the workspace volume
#    (not all on one line/plane -- the degeneracy guard will refuse a
#    flat/collinear set):
python3 perception-calibration-gap/calibration/capture_marker_points.py \
    --image <captured_frame.png> \
    --intrinsics perception-calibration-gap/calibration/camera_intrinsics.yaml \
    --marker-sizes-cm 0:5.0,1:5.0,2:5.0
# 3. By hand, measure each marker's position in the ROBOT BASE FRAME
#    (tape measure / calipers from a known robot-base reference point)
#    and fill base_x/base_y/base_z into
#    perception-calibration-gap/calibration/point_pairs.csv (replacing
#    the "REPLACE" placeholders capture_marker_points.py left there).
# 4. Solve:
python3 perception-calibration-gap/calibration/compute_extrinsic_calibration.py \
    --points perception-calibration-gap/calibration/point_pairs.csv \
    --output perception-calibration-gap/calibration/T_base_from_camera_optical.yaml
```
**Exit criterion**: the script prints held-out RMS < 2cm and SAVES the
YAML (if it refuses -- degenerate points, units mismatch, or residual
too high -- follow its printed error message and re-capture, don't
override the refusal).

## 4. Fill the grasp lookup table (~30-60 min, iterative)
For a spread of table positions covering your workspace: teleoperate the
arm to a good grasp pose over that (x, y), record the resulting joint
angles, add a `bins` entry to a copy of
`perception-calibration-gap/grasp/lookup_table_config_template.yaml`
(rename to `lookup_table_config.yaml`). More bins = better coverage;
start with ~9-12 spanning a 3x3-ish grid over the reachable tabletop
area.
**Exit criterion**: every bin's joint angles are within the config's
`joint_limits` (the executor will refuse IK_FAILURE otherwise) and a
dry-run test point lands on the bin you expect:
```bash
python3 perception-calibration-gap/grasp/lookup_table_executor.py \
    --grasp-point-xyz-camera "0.05,0.02,0.45" \
    --extrinsic perception-calibration-gap/calibration/T_base_from_camera_optical.yaml \
    --lookup-table perception-calibration-gap/grasp/lookup_table_config.yaml
```
(dry-run only — this never sends a real command; `--i-am-supervising`
is required for real motion and is a decision only you make, never
scripted).

## 5. Three-consecutive-reliable-grasps gate (~15-30 min)
With the arm under YOUR direct supervision (not autonomous), manually
trigger the lookup-table path (with `--i-am-supervising` set by you, at
the terminal, watching the arm) on 3 different bins in a row.
**Exit criterion**: all 3 grasps succeed (object picked up, no collision,
no joint-limit fault) before trusting the lookup table for real trials.
If any fails, re-teleop that bin's joint angles and retry.

## 6. Checkpoint 1: ground-truth marker validation scenes (~45-90 min)
Stage 5-8 known-geometry scenes on the apple (per
`perception-calibration-gap/checkpoint1_validation_scenes.py`'s design
methodology — occluder position/height, hand-calc formula already
documented there) plus 2-3 spot-check scenes per additional object.
Measure and fill real values into
`perception-calibration-gap/validation_scenes.csv` (replacing the
placeholder template rows — `hand_calc_validation.py` will REFUSE to run
on the template as-is, by design, until you do this).

| Scene | Object | Occluder present? | Measured half-extents / positions | Your notes |
|---|---|---|---|---|
| 1 | apple | no | | |
| 2 | apple | yes, partial | | |
| 3 | apple | yes, behind (depth-order check) | | |
| 4 | apple | yes, partial (different position) | | |
| 5 | apple | yes, partial (different height) | | |
| 6 | glass | spot-check | | |
| 7 | purse | spot-check | | |

Run:
```bash
python3 perception-calibration-gap/hand_calc_validation.py
```
**Exit criterion**: MAE < 5 percentage points (the script prints
PASS/FAIL directly). If it fails, the script's own printed guidance
lists the usual culprits (camera extrinsic drift, marker size error,
occluder height).

## 7. Real-depth PRG signal capture (the ONLY real-depth test of any PRG
signal, ~60-90 min) — see `explore/REAL_CAPTURE_PROTOCOL_V2.md` in full
Stage the 14 scenes described there (Contact/Support/Separate/near-contact
combinations of apple/purse/glass), measuring true gaps with calipers/
ruler as ground truth. Capture 5 RGB+depth frame pairs per scene.
```bash
python3 explore/scripts/e5_analyze_real_capture_gaps.py \
    --capture-dir real_capture_v2_YYYYMMDD/ \
    --bbox-csv real_capture_v2_YYYYMMDD/bbox.csv
```
**Exit criterion**: not a pass/fail gate — this is the actual sim-to-real
test. Report the AUROC/error numbers honestly, compare against
`explore/E3_FINDINGS.md`'s simulation numbers (C1=0.457, C2=0.486,
C3=0.463 AUROC), and note plainly that every result here is still
`mask_is_bbox_placeholder=True` (no real segmenter exists yet).

## 8. YOLO image capture (~90-120 min, spread across sessions ideally)
Per object (apple, purse, glass), capture real images across varied:
poses, lighting, backgrounds, occlusion levels, distances. Target at
least 50-100 images per object for a first real fine-tune attempt (more
is better; `train_yolo_scaffold.py`'s synthetic-dataset code path shows
the exact label format expected). Auto-label using markers (per T5's
"oracle-camera + marker auto-labeling" design, still to be built once
real images exist — flagged as a genuine next step, not yet implemented
against real images since none exist).
**Exit criterion**: per-object image count logged; `per_object_gate.py`'s
thresholds (currently placeholders at precision/recall/AP50 > 0.5) should
be revisited once real training is attempted — they were chosen to
exercise the gate's PASS/FAIL logic on synthetic data, not as a
considered real-detector quality bar.

## 9. First live smoke trials (~30-60 min)
Once steps 1-6 all pass: run `perception-calibration-gap/run_trial.py`
against the real hardware path (not the MuJoCo mock), starting with a
handful of trials, arm under direct supervision. Confirm
`perception-calibration-gap/analysis/data_quality_report.py` runs
cleanly against the resulting log:
```bash
python3 perception-calibration-gap/analysis/data_quality_report.py \
    --log perception-calibration-gap/mock-pipeline/data/<your_real_log>.jsonl
```
**Exit criterion**: no session/object confounding flagged, no controller-
freeze violation, `pose_source` populated correctly on every trial.
