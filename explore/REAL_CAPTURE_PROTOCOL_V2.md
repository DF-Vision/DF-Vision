# Real-Hardware Capture Protocol V2 — Measured-Gap Ground Truth

Supersedes/extends `retrain/REAL_CAPTURE_PROTOCOL.md` (still valid for
its original purpose — relation-classifier accuracy on real frames). This
version adds what E1/E3 specifically need: **physically measured ground-
truth gaps** (calipers or a ruler), since E1/E3's whole analysis turns on
comparing an estimated gap against a true one, and in simulation "true"
came free from the SDF — on real hardware, it has to be measured by hand.
Who runs this: Manasi, arm off, camera only. Estimated time: 45-75
minutes (a bit longer than V1, because of the caliper measurements).

## Objects and scenes

Same 3 objects as `retrain/REAL_CAPTURE_PROTOCOL.md` (apple, purse,
glass) plus one flat, rigid neutral prop for Support scenes (a book or a
flat box — anything with a clean, flat top surface makes the "stacked"
geometry unambiguous to measure).

For EVERY scene, stage it, then **measure the true minimum gap between
the two objects' nearest surfaces with a ruler or calipers BEFORE
capturing any frames** — write the number down immediately, not from
memory later. Calipers preferred for anything under ~5mm (a ruler can't
resolve sub-mm gaps reliably); a ruler is fine for anything over ~1cm.

| # | Scene | Target relation | What to measure |
|---|---|---|---|
| 1 | apple touching purse, side by side | Contact | gap ≈ 0, measure any visible sliver (calipers) |
| 2 | apple 2mm from purse (matchstick/coin as a spacer, then removed before capture) | Separate (near-contact) | the spacer's actual thickness, measured separately with calipers |
| 3 | apple 5mm from purse (same spacer method) | Separate (near-contact) | spacer thickness |
| 4 | apple 15mm from purse | Separate | ruler |
| 5 | apple 50mm from purse | Separate | ruler |
| 6 | apple 150mm from purse | Separate | ruler |
| 7 | apple placed on the flat prop, prop touching purse | Support (apple/prop), Contact (prop/purse) | gap ≈ 0 for both pairs, calipers |
| 8-14 | Repeat 1-7 with glass in place of apple | (same pattern) | (same) |

This gives 14 scenes total spanning true gap from 0 to 150mm, with 6
genuinely different near-contact gaps (2mm, 5mm) that no prior session's
simulation data had a way to produce (the ray-cast/noise-model synthetic
scenes only had whatever gaps the 6 authored SDF worlds happened to
contain).

**Spacer method for 2mm/5mm gaps**: place a known, calipers-measured
spacer object (a coin, a matchstick, a stack of index cards — whatever's
on hand) between the two objects to set the gap, measure the spacer's
thickness with calipers, remove the spacer, capture frames without it in
frame. Write the measured spacer thickness as the gap in
`ground_truth.csv` (not a guess).

## Frames per scene

5 RGB + depth frame pairs per scene, matching V1's protocol (a few
seconds apart, small natural jitter is fine/useful).

## Folder structure and naming

```
real_capture_v2_YYYYMMDD/
  scene01_apple_purse_contact/
    rgb_1.png ... rgb_5.png
    depth_1.npy ... depth_5.npy
  scene02_apple_purse_2mm/
    ...
  ground_truth.csv
  bbox.csv
```

`ground_truth.csv` template:
```csv
scene_folder,object_a,object_b,true_relation,measured_gap_m,measurement_tool,notes
scene01_apple_purse_contact,apple,purse,Contact,0.0000,calipers,visually flush; calipers read 0.0mm at closest point
scene02_apple_purse_2mm,apple,purse,Separate,0.0021,calipers,spacer was a US nickel measured at 2.1mm
scene04_apple_purse_15mm,apple,purse,Separate,0.0150,ruler,
scene07_apple_prop_purse,apple,prop,Support,0.0000,calipers,apple resting flat on prop's top surface
scene07_apple_prop_purse,prop,purse,Contact,0.0000,calipers,prop's edge touching purse
```
(Scene 7's two pairs go on two rows, same `scene_folder`.)

`bbox.csv` template (same format as `retrain/scripts/r8_analyze_real_capture.py`
already expects): `scene_folder,object,x0,y0,x1,y1` — one row per object
per scene, hand-drawn bounding boxes (any simple image viewer's pixel
coordinates work; a real segmenter would be better but per `SCHEMA.md `
section 5, none exists yet for real hardware).

## Capture-time scoring command

Run this once per scene right after capturing its frames (lets you
sanity-check on the spot rather than discovering a bad frame back at the
desk):
```bash
python3 explore/scripts/e5_analyze_real_capture_gaps.py \
    --capture-dir real_capture_v2_YYYYMMDD/ \
    --bbox-csv real_capture_v2_YYYYMMDD/bbox.csv \
    --scene scene01_apple_purse_contact
```

## Full analysis command (after all scenes are captured)

```bash
python3 explore/scripts/e5_analyze_real_capture_gaps.py \
    --capture-dir real_capture_v2_YYYYMMDD/ \
    --bbox-csv real_capture_v2_YYYYMMDD/bbox.csv
```
Computes, per scene and pooled: C1/C2/C3 candidate scores (in-memory
config override, `relation_classifier/prg_edge_error_config.json` on
disk is never touched), each candidate's internal `estimated_gap` vs the
hand-measured `measured_gap_m`, error, and whether error exceeds 1cm
(matching E3's own threshold) — i.e., criterion (c) re-run on REAL
measured gaps instead of the SDF oracle. Also reports AUROC for flagging
>1cm errors, exactly like `explore/scripts/e3_evaluate_candidates.py`
did in simulation, so the two numbers are directly comparable.

**This is the actual sim-to-real validity test for E3's candidates** —
the one thing this and no prior session could do, because it needs a
real depth camera and real physical measurements. Given E3's simulation
result was a decisive negative (no candidate cleared any of the
validity/sensitivity gates), the most likely realistic outcome of running
this for real is a similarly negative result — but that is exactly the
kind of prediction that must be checked, not assumed, before being
written down as settled.

## What this does NOT settle

Per rule 2: nothing from this protocol is evidence for C1/C1-ext/C4/C4-ext,
and nothing here validates failure prediction — that needs real grasp
trials with the arm, which this protocol explicitly does not include
(arm stays off throughout).
