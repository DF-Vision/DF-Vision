"""
hand_calc_validation.py
========================
Phase 1 Checkpoint 1 tool: validates occlusion_estimator.py's compute_true_visibility()
against an INDEPENDENT, hand-checkable reference calculation, on real physically-measured
scenes.

WHY THIS EXISTS (read before running):
occlusion_estimator.py uses a disclosed simplification: 2D-rectangle (AABB) projection +
depth-ordering, not full 3D ray-casting. This script implements that SAME geometric model
(pinhole projection of an axis-aligned bounding box -> 2D image-plane rectangle -> rectangle
overlap), independently, from scratch, using only numpy + basic pinhole-camera math. It is
NOT allowed to import anything from occlusion_estimator.py -- that would defeat the whole
point (Gap must be independent of the module it's checking).

The output MAE between this script's numbers and occlusion_estimator.py's numbers is your
Checkpoint 1 pass/fail metric (tolerance target: MAE < 5pp, i.e. < 0.05 on a 0-1 scale).

HOW TO USE:
1. Physically measure your scene: object dimensions (from generate_markers.py's manifest --
   apple 3.5cm, tennis_ball 3.5cm, purse 5.5cm, glass 4.5cm, remote 4.5cm marker sizes are
   already known; you additionally need each object's own bounding-box half-extents, and the
   occluder's half-extents), and marker-detected poses (position + orientation) for target,
   occluder, and camera, in a common world frame.
2. Fill in one row per scene in validation_scenes.csv (template written by this script on
   first run if the file doesn't exist).
3. Run this script. It will:
   a) compute an independent hand-checkable expected visibility per scene (this script's
      own reference math)
   b) call your REAL occlusion_estimator.compute_true_visibility() on the same inputs
   c) report per-scene values, MAE, and a pass/fail verdict against tolerance
   d) save a y=x scatter plot (pipeline vs hand-calc) as validation_scatter.png

WORKED EXAMPLE (do this by hand once, on paper/calculator, before trusting either the script
or the pipeline -- this is the actual "hand calculation" Phase 1 requires):

  Say the camera looks straight down the +Z axis. A target object's AABB has world-frame
  corners spanning x in [-0.03, 0.03], y in [-0.03, 0.03] at depth z=0.50m (a 6cm cube,
  matching a roughly apple-sized object). With focal length fx=fy=600px, principal point
  (cx,cy)=(320,240), the projected image-plane rectangle corners are:

      u = fx * (x/z) + cx,   v = fy * (y/z) + cy

  For the corner (x=-0.03, y=-0.03, z=0.50):
      u = 600*(-0.03/0.50) + 320 = 600*(-0.06) + 320 = -36 + 320 = 284
      v = 600*(-0.03/0.50) + 240 = 284 (symmetric in this example)
  For the corner (x=0.03, y=0.03, z=0.50):
      u = 600*(0.06) + 320 = 356
      v = 356

  So the target's projected rectangle spans u in [284,356], v in [284,356] -- an image-plane
  box 72x72 px, area = 5184 px^2.

  If an occluder's projected rectangle is, say, u in [300,340], v in [270,400] (a vertical
  bar crossing in front), the OVERLAP rectangle is:
      u_overlap = [max(284,300), min(356,340)] = [300,340]  -> width 40
      v_overlap = [max(284,270), min(356,400)] = [284,356]  -> height 72
      overlap_area = 40*72 = 2880 px^2

  visible_fraction = 1 - (overlap_area / target_area) = 1 - (2880/5184) = 1 - 0.5556 = 0.444

  That 0.444 is a number you can verify with a calculator in two minutes. This script
  automates exactly this math (plus depth-ordering: an occluder BEHIND the target, i.e.
  larger z, does not occlude it at all -- overlap is only counted if occluder_z < target_z).

DEPTH-ORDERING RULE (must match occlusion_estimator.py's disclosed behavior):
  If occluder's centroid z >= target's centroid z (occluder is at or behind the target),
  the occluder contributes ZERO occlusion regardless of 2D overlap. This is the same
  simplification occlusion_estimator.py uses (per its docstring) -- if your pipeline's
  numbers diverge from this script specifically in occluder-behind-target scenes, that's
  a real bug to chase, not a rounding difference.
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Same convention as run_trial.py: occlusion_estimator.py lives in gazebo/,
# not the project root, so it isn't importable by default. This makes the
# script location-independent -- it works whether you run it from the
# project root or anywhere else, as long as it sits at the project root
# itself (see placement note in the module docstring below).
sys.path.insert(0, str(Path(__file__).parent / "gazebo"))

SCENES_CSV = "validation_scenes.csv"
REPORT_CSV = "validation_report.csv"
SCATTER_PNG = "validation_scatter.png"
MAE_TOLERANCE = 0.05  # 5pp, per master reference Checkpoint 1 pass criterion

CSV_HEADER = [
    "scene_id", "object_id",
    # target object AABB half-extents (meters), and world-frame centroid position
    "target_half_x", "target_half_y", "target_half_z",
    "target_cx", "target_cy", "target_cz",
    # occluder AABB half-extents and centroid position (leave occluder_* blank / 0 half-extents
    # for a no-occluder baseline scene)
    "occ_half_x", "occ_half_y", "occ_half_z",
    "occ_cx", "occ_cy", "occ_cz",
    # camera intrinsics (assumes camera looking down +Z in its own frame; if your camera has
    # a nontrivial world-frame rotation, pre-rotate the target/occluder positions into camera
    # frame before filling this CSV -- keep that transform written down in your lab notebook
    # so it's independently checkable too)
    "fx", "fy", "cx_px", "cy_px",
    # ground-truth note field -- physical description so a labmate can redo the measurement
    "notes",
]


def write_template_if_missing(path: str) -> None:
    if os.path.exists(path):
        return
    print(f"[info] {path} not found -- writing a starter template with 3 example rows. "
          f"Replace these with your real measured scenes (aim for 5-8 on the primary object "
          f"'apple', plus 2-3 spot-check rows per additional object).")
    example_rows = [
        {
            "scene_id": "apple_baseline_noocc", "object_id": "apple",
            "target_half_x": 0.035, "target_half_y": 0.035, "target_half_z": 0.035,
            "target_cx": 0.0, "target_cy": 0.0, "target_cz": 0.50,
            "occ_half_x": 0, "occ_half_y": 0, "occ_half_z": 0,
            "occ_cx": 0, "occ_cy": 0, "occ_cz": 0,
            "fx": 600, "fy": 600, "cx_px": 320, "cy_px": 240,
            "notes": "No occluder present -- expected visibility should be 1.0 exactly. "
                     "REPLACE with your real measured apple half-extent and camera calibration.",
        },
        {
            "scene_id": "apple_partial_occ_01", "object_id": "apple",
            "target_half_x": 0.035, "target_half_y": 0.035, "target_half_z": 0.035,
            "target_cx": 0.0, "target_cy": 0.0, "target_cz": 0.50,
            "occ_half_x": 0.02, "occ_half_y": 0.06, "occ_half_z": 0.10,
            "occ_cx": 0.01, "occ_cy": 0.0, "occ_cz": 0.35,
            "fx": 600, "fy": 600, "cx_px": 320, "cy_px": 240,
            "notes": "Example partial occlusion (occluder in front, z=0.35 < target z=0.50). "
                     "REPLACE with real marker-detected poses.",
        },
        {
            "scene_id": "apple_occluder_behind", "object_id": "apple",
            "target_half_x": 0.035, "target_half_y": 0.035, "target_half_z": 0.035,
            "target_cx": 0.0, "target_cy": 0.0, "target_cz": 0.50,
            "occ_half_x": 0.05, "occ_half_y": 0.05, "occ_half_z": 0.05,
            "occ_cx": 0.0, "occ_cy": 0.0, "occ_cz": 0.80,
            "fx": 600, "fy": 600, "cx_px": 320, "cy_px": 240,
            "notes": "Depth-ordering check: occluder is BEHIND target (z=0.80 > 0.50) despite "
                     "large 2D overlap. Expected visibility should be 1.0 regardless of overlap "
                     "-- this row specifically tests the depth-ordering rule, not the projection "
                     "math.",
        },
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in example_rows:
            writer.writerow(row)


def refuse_if_still_template(path: str, template_writer=write_template_if_missing) -> None:
    """T3 fix (project/overnight-hardening session): the existing
    'no data rows' check below does NOT catch the actual state
    validation_scenes.csv is in -- the auto-written template has 3
    NON-EMPTY placeholder rows (see write_template_if_missing above),
    each with a `notes` field containing the literal marker text
    "REPLACE with" and camera intrinsics/geometry that are illustrative
    only. Without this guard, running this script against the untouched
    template would compute a (meaningless) MAE and could be mistaken for
    a real Checkpoint 1 pass. Refuses if EITHER condition holds:
      (a) the file is byte-identical to a freshly-generated template, or
      (b) any row's notes field contains the literal "REPLACE with".
    Never modifies validation_scenes.csv itself."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        template_writer(tmp_path)
        with open(path, "rb") as f_real, open(tmp_path, "rb") as f_template:
            if f_real.read() == f_template.read():
                raise ValueError(
                    f"{path} is byte-identical to the auto-generated template -- "
                    f"this is placeholder data, not measured scenes. REFUSING to "
                    f"treat it as a Checkpoint 1 result. Fill in real measured rows "
                    f"and re-run."
                )
    finally:
        os.remove(tmp_path)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            notes = row.get("notes", "") or ""
            if "REPLACE with" in notes:
                raise ValueError(
                    f"{path} row {row.get('scene_id', '?')!r} still contains the "
                    f"template's literal \"REPLACE with\" marker text in its notes "
                    f"field -- this row has not been filled in with real measured "
                    f"data. REFUSING to treat this file as a Checkpoint 1 result "
                    f"until every placeholder row is replaced."
                )


@dataclass
class AABBScene:
    scene_id: str
    object_id: str
    target_half: np.ndarray   # (3,)
    target_c: np.ndarray      # (3,)
    occ_half: np.ndarray      # (3,)
    occ_c: np.ndarray         # (3,)
    fx: float
    fy: float
    cx_px: float
    cy_px: float
    notes: str

    @property
    def has_occluder(self) -> bool:
        return bool(np.any(self.occ_half > 0))


def load_scenes(path: str) -> list:
    scenes = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenes.append(AABBScene(
                scene_id=row["scene_id"],
                object_id=row["object_id"],
                target_half=np.array([float(row["target_half_x"]),
                                       float(row["target_half_y"]),
                                       float(row["target_half_z"])]),
                target_c=np.array([float(row["target_cx"]),
                                    float(row["target_cy"]),
                                    float(row["target_cz"])]),
                occ_half=np.array([float(row["occ_half_x"]),
                                    float(row["occ_half_y"]),
                                    float(row["occ_half_z"])]),
                occ_c=np.array([float(row["occ_cx"]),
                                 float(row["occ_cy"]),
                                 float(row["occ_cz"])]),
                fx=float(row["fx"]), fy=float(row["fy"]),
                cx_px=float(row["cx_px"]), cy_px=float(row["cy_px"]),
                notes=row.get("notes", ""),
            ))
    return scenes


def project_aabb_to_rect(center: np.ndarray, half_extent: np.ndarray,
                          fx: float, fy: float, cx: float, cy: float):
    """
    Project an axis-aligned bounding box's 8 corners through a pinhole camera model
    (camera looking down +Z, standard u = fx*x/z + cx, v = fy*y/z + cy), and return the
    enclosing 2D rectangle (u_min, u_max, v_min, v_max) in the image plane.

    This deliberately mirrors the disclosed 2D-rect simplification in occlusion_estimator.py
    -- it is NOT full 3D ray-casting, on purpose, so this script's numbers are checking the
    SAME model the pipeline claims to implement, not a stricter one.
    """
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = center[None, :] + signs * half_extent[None, :]  # (8,3)

    z = corners[:, 2]
    if np.any(z <= 0):
        raise ValueError(
            f"Object has a corner at or behind the camera plane (z<=0): {corners}. "
            f"Check your measured centroid/half-extent inputs -- this is almost always a "
            f"units mistake (e.g. mm entered where meters expected) or a genuinely "
            f"misconfigured scene."
        )

    u = fx * (corners[:, 0] / z) + cx
    v = fy * (corners[:, 1] / z) + cy

    return u.min(), u.max(), v.min(), v.max()


def rect_overlap_area(rect_a, rect_b) -> float:
    """rect = (u_min, u_max, v_min, v_max). Returns 0 if no overlap."""
    u_lo = max(rect_a[0], rect_b[0])
    u_hi = min(rect_a[1], rect_b[1])
    v_lo = max(rect_a[2], rect_b[2])
    v_hi = min(rect_a[3], rect_b[3])
    width = max(0.0, u_hi - u_lo)
    height = max(0.0, v_hi - v_lo)
    return width * height


def rect_area(rect) -> float:
    return max(0.0, rect[1] - rect[0]) * max(0.0, rect[3] - rect[2])


def hand_calc_visibility(scene: AABBScene) -> float:
    """
    Independent reference implementation: AABB projection + rectangle overlap +
    depth-ordering. This is the "hand calculation" -- every number here is reproducible
    with a calculator, per the worked example in this file's module docstring.
    """
    target_rect = project_aabb_to_rect(
        scene.target_c, scene.target_half, scene.fx, scene.fy, scene.cx_px, scene.cy_px
    )
    target_area = rect_area(target_rect)
    if target_area <= 0:
        raise ValueError(f"Scene {scene.scene_id}: target projects to zero area -- check inputs.")

    if not scene.has_occluder:
        return 1.0

    # Depth-ordering rule: occluder only occludes if it is strictly in front of the target
    # (smaller z = closer to camera, assuming +Z is the viewing direction).
    if scene.occ_c[2] >= scene.target_c[2]:
        return 1.0

    occ_rect = project_aabb_to_rect(
        scene.occ_c, scene.occ_half, scene.fx, scene.fy, scene.cx_px, scene.cy_px
    )
    overlap = rect_overlap_area(target_rect, occ_rect)
    visible_fraction = 1.0 - (overlap / target_area)
    return max(0.0, min(1.0, visible_fraction))


def call_real_pipeline(scene: AABBScene):
    """
    Calls the REAL occlusion_estimator.compute_true_visibility(), matching its actual
    signature exactly:
        ObjectGeometry(width, height, depth, marker_to_object_offset=(0,0,0))
        Pose(x, y, z, yaw=0.0)
        compute_true_visibility(target_pose, target_geom, occluder_pose, occluder_geom,
                                 camera_intrinsics)

    NOTE on units: this CSV's target_half_x/y/z are HALF-extents (matching this script's
    own hand-calc convention), but ObjectGeometry takes FULL width/height/depth -- so we
    double them here. Don't skip this conversion, or every comparison will silently be off
    by a factor of 2 in size (which usually still "runs" without erroring, so it's an easy
    bug to miss -- check your first few scenes' pipeline_visibility values against the
    hand_calc column by eye before trusting the full MAE).

    occlusion_estimator.py's compute_true_visibility does NOT accept occ_geom/occ_pose as
    None for a no-occluder scene (unlike this script's own hand_calc_visibility, which
    short-circuits to 1.0). If you have a genuine no-occluder baseline row, pass a
    degenerate, far-away, zero-size occluder instead of None -- see the fallback below.
    """
    from occlusion_estimator import ObjectGeometry, Pose, compute_true_visibility

    target_geom = ObjectGeometry(
        width=2 * scene.target_half[0], height=2 * scene.target_half[1], depth=2 * scene.target_half[2]
    )
    target_pose = Pose(x=scene.target_c[0], y=scene.target_c[1], z=scene.target_c[2])

    if scene.has_occluder:
        occ_geom = ObjectGeometry(
            width=2 * scene.occ_half[0], height=2 * scene.occ_half[1], depth=2 * scene.occ_half[2]
        )
        occ_pose = Pose(x=scene.occ_c[0], y=scene.occ_c[1], z=scene.occ_c[2])
    else:
        occ_geom = ObjectGeometry(width=0.0, height=0.0, depth=0.0)
        occ_pose = Pose(x=0.0, y=0.0, z=scene.target_c[2] + 10.0)

    camera_intrinsics = {
        "fx": scene.fx, "fy": scene.fy, "cx": scene.cx_px, "cy": scene.cy_px,
    }

    return compute_true_visibility(target_pose, target_geom, occ_pose, occ_geom,
                                    camera_intrinsics)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", default=SCENES_CSV,
                         help="Path to validation_scenes.csv (created if missing)")
    parser.add_argument("--skip-pipeline", action="store_true",
                         help="Only run the hand-calc reference, don't import "
                              "occlusion_estimator.py (useful for sanity-checking this "
                              "script's own math before the real module is ready)")
    parser.add_argument("--tolerance", type=float, default=MAE_TOLERANCE,
                         help=f"MAE pass/fail tolerance, default {MAE_TOLERANCE}")
    args = parser.parse_args()

    write_template_if_missing(args.scenes)
    try:
        refuse_if_still_template(args.scenes)
    except ValueError as e:
        print(f"[REFUSED] {e}")
        sys.exit(1)
    scenes = load_scenes(args.scenes)
    if not scenes:
        print(f"[error] {args.scenes} has no data rows. Fill in real measured scenes and rerun.")
        sys.exit(1)

    results = []
    abs_errors = []
    for scene in scenes:
        try:
            hand_val = hand_calc_visibility(scene)
        except ValueError as e:
            print(f"[error] scene {scene.scene_id}: {e}")
            continue

        pipeline_val: Optional[float] = None
        error = None
        if not args.skip_pipeline:
            try:
                pipeline_val = call_real_pipeline(scene)
                error = abs(pipeline_val - hand_val)
                abs_errors.append(error)
            except ImportError as e:
                print(f"[error] Could not import occlusion_estimator.py: {e}\n"
                      f"        Run with --skip-pipeline to test only this script's own math, "
                      f"or fix your PYTHONPATH / run this from the same directory as your "
                      f"real modules.")
                sys.exit(1)
            except Exception as e:
                print(f"[error] scene {scene.scene_id}: real pipeline raised: {e}")

        results.append({
            "scene_id": scene.scene_id,
            "object_id": scene.object_id,
            "hand_calc_visibility": round(hand_val, 4),
            "pipeline_visibility": round(pipeline_val, 4) if pipeline_val is not None else "",
            "abs_error": round(error, 4) if error is not None else "",
            "notes": scene.notes,
        })

    # Write report CSV
    with open(REPORT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "object_id", "hand_calc_visibility",
                                                "pipeline_visibility", "abs_error", "notes"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n{'scene_id':<24}{'object':<12}{'hand_calc':<12}{'pipeline':<12}{'abs_err':<10}")
    for r in results:
        print(f"{r['scene_id']:<24}{r['object_id']:<12}{r['hand_calc_visibility']:<12}"
              f"{r['pipeline_visibility']!s:<12}{r['abs_error']!s:<10}")

    if abs_errors:
        mae = float(np.mean(abs_errors))
        max_err = float(np.max(abs_errors))
        verdict = "PASS" if mae < args.tolerance else "FAIL"
        print(f"\nMAE = {mae:.4f}   Max error = {max_err:.4f}   "
              f"Tolerance = {args.tolerance}   Verdict: {verdict}")
        if verdict == "FAIL":
            print("[!] MAE exceeds tolerance. Do NOT proceed to Phase 1.5 (YOLO training) "
                  "or beyond. Usual culprits (per master reference): camera extrinsic drift, "
                  "marker-size misconfiguration (physical vs. configured dimension), or a "
                  "units mismatch (mm vs. m) between this CSV and your real pipeline's inputs.")

        # Plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            hand_vals = [r["hand_calc_visibility"] for r in results if r["pipeline_visibility"] != ""]
            pipe_vals = [r["pipeline_visibility"] for r in results if r["pipeline_visibility"] != ""]
            plt.figure(figsize=(5, 5))
            plt.scatter(hand_vals, pipe_vals, color="tab:blue")
            plt.plot([0, 1], [0, 1], "k--", label="y = x")
            plt.xlabel("Hand-calculated visibility")
            plt.ylabel("Pipeline (occlusion_estimator.py) visibility")
            plt.title(f"Checkpoint 1 validation -- MAE={mae:.4f}")
            plt.legend()
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.tight_layout()
            plt.savefig(SCATTER_PNG, dpi=150)
            print(f"[info] Scatter plot saved to {SCATTER_PNG}")
        except ImportError:
            print("[info] matplotlib not available -- skipped scatter plot.")
    else:
        print("\n[info] Ran in --skip-pipeline mode (or all pipeline calls failed) -- "
              "no MAE computed. This only validates this script's own hand-calc math, "
              "not the real pipeline.")

    print(f"\nFull report written to {REPORT_CSV}")


if __name__ == "__main__":
    main()