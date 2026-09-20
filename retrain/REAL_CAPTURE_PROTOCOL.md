# Real-Hardware Capture Protocol — Sim-to-Real Test for the Relation Classifier

**Who runs this**: Manasi, by hand. Arm powered OFF throughout — camera
only. Estimated time: 30-60 minutes. This is the one piece of validation
no autonomous session can do; everything else in this retrain effort has
been simulation.

## What this collects

Staged Contact / Support / Separate / near-contact scenes for the 3
real-hardware objects named in `INTERFACE.md`'s Option A (apple, purse,
glass — note: `PREREG.md` itself lists 5 objects including a
remote/stapler substitution; `INTERFACE.md` says 3. This protocol uses
the 3-object set since that's what `INTERFACE.md` states is the actual
real-hardware plan, but this is exactly the unresolved object-count
conflict flagged in `PREREG_AMENDMENTS_PROPOSED.md` — if that gets
resolved before you run this, use whichever object set is current).

## Scenes to stage (minimum set — more is better, time permitting)

For each pair of the 3 objects (apple/purse, apple/glass, purse/glass)
and each object against a **fourth, neutral prop** (anything already on
the table — a mug, a book, whatever's on hand) to get a genuine
"different object identities than used anywhere in the sim dataset" test:

| # | Scene | Relation (ground truth, YOU set this up and know it) |
|---|---|---|
| 1 | apple resting directly against purse, touching | Contact |
| 2 | apple resting on top of purse (purse flat, apple placed on it) | Support |
| 3 | apple and purse ~15cm apart, both visible | Separate |
| 4 | apple ~1-2cm from purse, clearly NOT touching (near-contact) | Separate (near) |
| 5 | glass resting against apple, touching | Contact |
| 6 | glass placed on top of a flat book/box, book touching apple | Support (book/apple) — pick whatever's staged |
| 7 | glass and apple ~15cm apart | Separate |
| 8 | purse and glass touching, side by side | Contact |
| 9 | purse and glass stacked (glass on purse, if stable — use caution, don't let it fall) | Support |
| 10 | purse and glass ~15cm apart | Separate |

Repeat each scene at 2-3 camera positions/angles if time allows (doubles
or triples the effective test set for free). At minimum, get all 10 base
scenes from one consistent camera position first — that's the priority.

## Frames per scene

**5 RGB + depth frame pairs per scene**, a couple of seconds apart (lets
you catch and discard a blurry/motion frame if the camera moved slightly
between shots). You don't need to hold perfectly still — small natural
camera-jitter across the 5 frames is fine and arguably useful (adds a
tiny bit of realistic pose variation "for free").

## Where to save, naming

```
real_capture_YYYYMMDD/
  scene01_apple_purse_contact/
    rgb_1.png rgb_2.png rgb_3.png rgb_4.png rgb_5.png
    depth_1.npy depth_2.npy depth_3.npy depth_4.npy depth_5.npy
  scene02_apple_purse_support/
    ...
  ground_truth.csv   <- fill in by hand, template below
```

`ground_truth.csv` template (one row per scene folder):
```csv
scene_folder,object_a,object_b,true_relation,notes
scene01_apple_purse_contact,apple,purse,Contact,
scene02_apple_purse_support,apple,purse,Support,apple placed flat on purse's side
scene03_apple_purse_separate,apple,purse,Separate,~15cm apart
scene04_apple_purse_near,apple,purse,Separate,~1-2cm gap intentionally not touching
```
`true_relation` must be exactly one of `Contact`/`Support`/`Separate`
(matches the model's output vocabulary exactly, case-sensitive).

## Capture command

This project's existing capture path is camera-specific (Orbbec Gemini 2
per `PROJECT_STATUS.md`/`PREREG.md`) and this session did not have that
camera connected to test against, so the exact capture script may need a
small adjustment on the day — check first whether
`perception-calibration-gap/real_camera/capture_and_gap.py` (referenced
in `PROJECT_STATUS.md` §3's `masks_are_bbox_placeholder` fix) or a
similar existing entry point already does RGB+depth capture; if so, run
it once per frame with the scene name as an argument, saving into the
folder structure above. If no ready-made single-shot capture script
exists, a minimal fallback (pyrealsense2/OpenNI2-equivalent for the
Orbbec SDK, whichever this project already has installed) is:
```python
# illustrative -- adapt to whatever camera SDK is actually installed;
# this project's own capture_and_gap.py is the preferred, already-tested
# starting point if it exists and runs
import cv2, numpy as np
# ... camera.get_frames() -> rgb (H,W,3 uint8), depth (H,W float32 metres) ...
cv2.imwrite(f"{scene_dir}/rgb_{i}.png", rgb)
np.save(f"{scene_dir}/depth_{i}.npy", depth)
```

## Analysis command (run this after capture, from the project root)

A ready-made analysis script for this exact folder structure:
`retrain/scripts/r8_analyze_real_capture.py` (created this session,
untested against real data since none exists yet — test it on one dummy
scene folder with synthetic placeholder files first if anything looks
off). Usage:
```bash
python3 retrain/scripts/r8_analyze_real_capture.py --capture-dir real_capture_YYYYMMDD/ --model-version legacy
python3 retrain/scripts/r8_analyze_real_capture.py --capture-dir real_capture_YYYYMMDD/ --model-version v2
```
Computes and prints, for each `--model-version`: per-scene predictions,
overall accuracy and confusion matrix against `ground_truth.csv`,
`prg_edge_error`'s value per scene and its AUROC for flagging wrong
predictions (G6, now on REAL data for the first time). Needs
`all_object_masks` for each scene — the script uses a simple manual
bounding-box-to-mask fallback (`mask_is_bbox_placeholder=True`, matching
`SCHEMA.md`'s own documented real-hardware fallback) since no real
segmentation model exists yet per `SCHEMA.md` §5 — flags every result
with that caveat, since `SCHEMA.md` itself says bbox-placeholder edges
are wiring-validation only, not reportable numbers. If a real segmenter
exists by the time you run this, swap it in and re-run for a clean
(non-placeholder) result.

## What this settles

This is the actual sim-to-real generalization test the paper needs and
the piece neither this nor any prior autonomous session could produce
(no camera, no arm, no real objects available in this environment). Once
run, compare its balanced accuracy / confusion matrix against
`retrain/models/r3_training_summary.txt`'s simulation CV numbers and
`retrain/r5_final_test_SIM.md`'s held-out-world numbers — a large gap
would indicate the sim-to-real domain shift is the dominant remaining
problem, not the model architecture or label quality.
