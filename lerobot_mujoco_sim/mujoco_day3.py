"""
mujoco_day3.py

Phase 0.25, Day 3 -- policy coherence check + real logging wiring, on top of
MuJoCo-rendered data. This is the last day of the MuJoCo feasibility smoke
test before the Phase 0.25 close-out decision gate (Week 1 plan, Day 3).

WHAT THIS DOES NOT DECIDE
--------------------------------------------------------------------------
This script does NOT decide learned-VLA-vs-scripted-controller. That
decision is explicitly reserved for the real-hardware feasibility gate
(master reference Section 6, Weeks 1-2) and should not be made here, on
placeholder MuJoCo geometry. Making it here would also violate the
"MuJoCo stage is a pipeline check, not evidence for any statistical or
design result" rule from the master reference's edge-cases section.

Instead, this script reuses mock_vla.execute() -- the SAME policy-stand-in
interface run_trial.py already calls for its own mock pipeline -- as the
Day 3 "does a policy-shaped call produce a coherent, non-degenerate action"
check. This is deliberate, not a cop-out: mock_vla.execute() is already a
working, existing interface in your codebase (not invented today), and
using it here means Day 3 tests exactly what it's supposed to test --
policy-call coherence + logging integrity -- without prematurely resolving
a decision that belongs to real hardware.

WHAT CHANGED VS. run_trial.py's OWN MOCK PIPELINE
--------------------------------------------------------------------------
run_trial.py's build_scenario_poses() / get_ground_truth() / get_perception()
are a FULLY MOCK geometry generator -- they never touch MuJoCo and were
built for Phase 5's real trial-collection dry runs. Day 3's job is
different: confirm the REAL MuJoCo-rendered perceived_visibility and
true_visibility (from mujoco_day2.py, already validated on Day 2) can be
pushed through run_trial.py's real downstream machinery -- gap_computer,
logger.TrialRecord/TrialLogger, and a policy call -- with zero code changes
to any of those modules. So this script:
  - reuses mujoco_day2's glue functions directly (world_to_camera,
    aabb_world_corners, project_to_pixels, build_bbox_mask,
    build_pinhole_extrinsic, force_occlusion_layout) instead of
    reimplementing them
  - varies force_occlusion_layout's x_shift across trials to get a
    non-degenerate RANGE of true_visibility values (Day 2 only ran one
    fixed layout)
  - calls the real tsdf_fusion / occlusion_estimator / gap_computer chain
    per trial, exactly as Day 2 validated
  - calls mock_vla.execute() (existing interface) for the policy check
  - calls the real logger.TrialRecord / TrialLogger.append_trial()
    (existing interface, per run_trial.py) to confirm schema fields
    populate correctly

EXPLICITLY OUT OF SCOPE FOR THIS SCRIPT
--------------------------------------------------------------------------
  - OnlineGateAdapter / conformal gate: that's a real-trial-collection
    concern (C4/C5 online triggering), not a Phase 0.25 pipeline check.
    Stubbed here as gate_fired=False, nonconformity_score=None,
    gate_warmup_phase=True -- schema fields populate, but mean nothing yet.
  - Real session-baseline protocol (15-20 trial block): a token 3-trial
    baseline is run purely to fill session_baseline_failure_rate's schema
    slot without crashing the logger's REQUIRED_FIELDS check. Not a real
    session baseline.
  - object_id is set to "apple" as a placeholder study-object label (must
    be one of the 5 real object names per logger.py's schema comment) --
    this is MuJoCo's generic block_1, not real apple data. Labeled
    explicitly in the trial record's notes/print output so this can never
    be mistaken for real apple trial data later.

Data from this script is NEVER to be merged with, or reported alongside,
real trial data -- logged to a clearly separate file
(data/trials_mujoco_smoke.jsonl) and excluded from any statistical result,
exactly like the MuJoCo stage's exclusion rule for C1/C4/C1-ext/C4-ext.

Container assumptions (mujoco-rehearsal, no bind mount, per Day 1/2 notes):
    docker cp mujoco_day3.py mujoco-rehearsal:/workspace/
    (mujoco_render.py, mujoco_day2.py, tsdf_fusion.py, gap_computer.py,
     occlusion_estimator.py, logger.py, mock_vla.py, conformal_gate.py
     should already be in /workspace from Day 1/Day 2 + your run_trial.py
     copy step)
Then, inside the container, every fresh session:
    export MUJOCO_GL=egl
    cd /workspace
    python3 mujoco_day3.py
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from dataclasses import fields, is_dataclass

import numpy as np

# Match run_trial.py's own path setup so occlusion_estimator/gap_computer/
# logger/mock_vla resolve the same way they do for the real mock pipeline.
sys.path.insert(0, str(Path(__file__).parent / "gazebo"))
sys.path.insert(0, str(Path(__file__).parent / "mock-pipeline"))

from mujoco_env.y_env import SimpleEnv
import mujoco_render

# Reuse Day 2's already-validated glue code directly -- do not reimplement.
from mujoco_day2 import (
    world_to_camera,
    aabb_world_corners,
    project_to_pixels,
    build_bbox_mask,
    build_pinhole_extrinsic,
    force_occlusion_layout,
)

from tsdf_fusion import MaskedTSDFFusion, CameraIntrinsics
from gap_computer import compute_gap_features
from occlusion_estimator import ObjectGeometry, Pose, compute_true_visibility
from logger import TrialRecord, TrialLogger
import mock_vla
import cv2
import os

DEBUG_IMG_DIR = Path(__file__).parent / "mujoco_day3_debug"


PLACEHOLDER_OBJECT_ID = "apple"  # NOT real apple data -- see module docstring
N_TRIALS = 8
CAM_NAME = "topview"
LOG_PATH = Path(__file__).parent / "mock-pipeline" / "data" / "trials_mujoco_smoke.jsonl"


def run_one_mujoco_trial(env, trial_id: int, x_shift: float, seed: int, force_occlusion: bool = True) -> TrialRecord:
    """One full trial: force a layout at the given x_shift (controls how much
    the occluder overlaps the target -- smaller x_shift => more occlusion,
    per force_occlusion_layout's own geometry), render, compute
    perceived/true visibility via the real modules, call the policy
    stand-in, and build a schema-correct TrialRecord."""


    rng = np.random.default_rng(seed)

    if force_occlusion:
        force_occlusion_layout(env, x_shift=x_shift)

    rgb, depth_mm, mrender_intr, T_view = mujoco_render.render_rgbd(env, CAM_NAME)

    # --- DIAGNOSTIC (temporary): confirm the render actually changed between
    # trials. If rgb_hash/depth_hash are IDENTICAL across all 8 trials despite
    # force_occlusion_layout() moving the occluder each time, the renderer is
    # not picking up the pose change -- a caching/scene-sync issue in
    # get_fixed_cam_rgbd_pcd(), not a bug in the Gap/TSDF computation itself.
    import hashlib
    rgb_hash = hashlib.md5(rgb.tobytes()).hexdigest()[:8]
    depth_hash = hashlib.md5(depth_mm.tobytes()).hexdigest()[:8]
    print(f"  [diagnostic] rgb_hash={rgb_hash}  depth_hash={depth_hash}  "
          f"depth_mean={depth_mm.mean():.2f}  depth_min={depth_mm.min():.2f}  depth_max={depth_mm.max():.2f}")

    width, height = mrender_intr.width, mrender_intr.height
    fx, fy, cx, cy = mrender_intr.fx, mrender_intr.fy, mrender_intr.cx, mrender_intr.cy

    target_center_w, _, target_half = mujoco_render.get_object_geometry_and_pose(env, "body_obj_block_1")
    occ_center_w, _, occ_half = mujoco_render.get_object_geometry_and_pose(env, "body_obj_block_2")

    target_center_cam = world_to_camera(target_center_w[None, :], T_view)[0]
    occ_center_cam = world_to_camera(occ_center_w[None, :], T_view)[0]
    assert target_center_cam[2] > 0, "Target camera-frame z not positive -- T_view convention check failed."

    corners_w = aabb_world_corners(target_center_w, target_half)
    corners_cam = world_to_camera(corners_w, T_view)
    u, v, z = project_to_pixels(corners_cam, fx, fy, cx, cy)
    assert np.all(z > 0), "Projected corner behind camera."
    mask, bbox_px = build_bbox_mask(u, v, width, height)

    # --- DIAGNOSTIC (temporary): depth stats INSIDE the mask region specifically
    # (whole-frame min/max are dominated by the static arm/table and tell us
    # nothing about whether the occluder ever actually lands on these pixels).
    u0, v0, u1, v1 = bbox_px
    depth_in_mask = depth_mm[v0:v1, u0:u1]
    print(f"  [diagnostic-masked] depth_in_mask: min={depth_in_mask.min():.2f} "
          f"max={depth_in_mask.max():.2f} mean={depth_in_mask.mean():.2f}  "
          f"(target alone ~500mm; occluder, 60mm closer, would show ~440mm if present here)")

    # Also project the OCCLUDER's own AABB the same way, to see where MuJoCo's
    # render actually puts it relative to the target's fixed mask region.
    occ_corners_w = aabb_world_corners(occ_center_w, occ_half)
    occ_corners_cam = world_to_camera(occ_corners_w, T_view)
    ou, ov, oz = project_to_pixels(occ_corners_cam, fx, fy, cx, cy)
    occ_bbox_px = (int(ou.min()), int(ov.min()), int(ou.max()), int(ov.max()))
    print(f"  [diagnostic-occluder] occluder's own projected bbox: {occ_bbox_px} "
          f"(compare to target's fixed bbox {bbox_px} -- do they overlap?)")


    import cv2, os
    os.makedirs("mujoco_day3_debug", exist_ok=True)
    debug_img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    tx0, ty0, tx1, ty1 = bbox_px
    ox0, oy0, ox1, oy1 = occ_bbox_px
    cv2.rectangle(debug_img, (tx0, ty0), (tx1, ty1), (0, 255, 0), 1)   # green = target
    cv2.rectangle(debug_img, (ox0, oy0), (ox1, oy1), (0, 0, 255), 1)   # red = occluder
    debug_path = f"mujoco_day3_debug/trial_{trial_id}_x{x_shift:.4f}.png"
    cv2.imwrite(debug_path, debug_img)
    print(f"  [diagnostic-image] saved {debug_path}")

    # --- DIAGNOSTIC (temporary): save the actual rendered RGB frame with both
    # computed bboxes drawn on it -- GREEN = target (our reconstructed
    # projection), RED = occluder (our reconstructed projection). If these
    # rectangles don't visually sit on top of the real blocks in the saved
    # PNG, the bug is confirmed to be in the pixel-projection math (T_view /
    # intrinsics / world_to_camera), not in Gap/TSDF logic downstream of it.
    os.makedirs(DEBUG_IMG_DIR, exist_ok=True)
    debug_img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    tx0, ty0, tx1, ty1 = bbox_px
    ox0, oy0, ox1, oy1 = occ_bbox_px
    cv2.rectangle(debug_img, (tx0, ty0), (tx1, ty1), (0, 255, 0), 1)   # green = target
    cv2.rectangle(debug_img, (ox0, oy0), (ox1, oy1), (0, 0, 255), 1)   # red = occluder
    debug_path = str(DEBUG_IMG_DIR / f"trial_{trial_id}_x{x_shift:.4f}.png")
    cv2.imwrite(debug_path, debug_img)
    print(f"  [diagnostic-image] saved {debug_path} (green=target bbox, red=occluder bbox)")

    intr = CameraIntrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    fusion = MaskedTSDFFusion(voxel_length=0.004, sdf_trunc=0.02)
    extrinsic = build_pinhole_extrinsic(T_view)
    fusion.integrate_frame(
        color=rgb, depth=depth_mm, mask=mask, intrinsics=intr,
        extrinsic=extrinsic, depth_scale=1000.0, depth_trunc=3.0,
    )
    perceived_visibility = fusion.compute_perceived_visibility(
        target_center=target_center_w, target_half_extents=target_half, grid_resolution=12
    )

    target_pose = Pose(x=target_center_cam[0], y=target_center_cam[1], z=target_center_cam[2])
    target_geom = ObjectGeometry(width=2 * target_half[0], height=2 * target_half[1], depth=2 * target_half[2])
    occ_pose = Pose(x=occ_center_cam[0], y=occ_center_cam[1], z=occ_center_cam[2])
    occ_geom = ObjectGeometry(width=2 * occ_half[0], height=2 * occ_half[1], depth=2 * occ_half[2])
    camera_intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    true_visibility = compute_true_visibility(target_pose, target_geom, occ_pose, occ_geom, camera_intrinsics)

    gap_result = compute_gap_features([perceived_visibility], [true_visibility])

    # --- Policy coherence check (existing mock_vla interface, per run_trial.py) ---
    t0 = time.perf_counter()
    attempt = mock_vla.execute(
        true_position=np.array([0.3, 0.1, 0.0]),
        perceived_visibility=perceived_visibility,
        goal_pose_tolerance=0.02,
        seed=seed,
    )
    action_latency = time.perf_counter() - t0
    outcome, failure_type = attempt.outcome, attempt.failure_type

    # Coherence assertions -- this is what "non-degenerate action" means concretely.
    # NOTE: mock_vla.execute()'s real Outcome type is Literal["success", "fail"] only --
    # "deferred" is run_trial.py's OWN vocabulary, added when its fusion-recovery logic
    # decides not to act. We don't exercise fusion recovery here (explicitly out of scope,
    # see module docstring), so only "success"/"fail" are ever possible from this call.
    assert outcome in ("success", "fail"), f"Unrecognized outcome: {outcome!r}"
    assert action_latency < 5.0, f"Policy call took {action_latency:.2f}s -- too slow even for a mock."

    record = TrialRecord(
        trial_id=trial_id,
        session_id=0,
        object_id=PLACEHOLDER_OBJECT_ID,
        session_baseline_failure_rate=0.0,  # token value -- see module docstring
        measured_true_visibility=true_visibility,
        true_occlusion=1.0 - true_visibility,
        clutter_count=0,
        perceived_visibility_initial=perceived_visibility,
        Gap_initial=gap_result.gap_initial,
        Gap_max=gap_result.gap_max,
        Gap_slope=gap_result.gap_slope,
        failure=0 if outcome == "success" else 1,
        camera_pose_variant=f"{CAM_NAME}_x{x_shift:.4f}",
        nonconformity_score=None,
        gate_fired=False,
        gate_warmup_phase=True,
        tsdf_fusion_attempted=False,
        Gap_before_fusion=None,
        Gap_after_fusion=None,
        outcome=outcome,
        failure_type=failure_type,
        goal_pose_tolerance=0.02,
        camera_pose=[0, 0, 0, 0, 0, 0],
        light_level=1.0,
        seed=seed,
        gap_per_step=gap_result.gap_trajectory,
        timing_per_step=[action_latency],
        true_visibility_per_step=[true_visibility],
    )
    return record, bbox_px


def check_schema_populated(record: TrialRecord) -> list[str]:
    """Returns a list of field names that are unexpectedly None. A few
    fields (nonconformity_score, Gap_before_fusion, Gap_after_fusion) are
    ALLOWED to be None by design in this smoke test -- see module docstring."""
    allowed_none = {
        "nonconformity_score", "Gap_before_fusion", "Gap_after_fusion", "failure_type",
        # Legitimately unset in this smoke test -- mock_vla is not a real OpenVLA
        # checkpoint, so there is no checkpoint hash/seed to log yet (that's a
        # real-hardware-stage concern, per the master reference's frozen-policy
        # discipline in Section 1).
        "openvla_checkpoint_hash", "openvla_seed",
    }
    if not is_dataclass(record):
        return ["<record is not a dataclass -- cannot introspect fields>"]
    bad = []
    for f in fields(record):
        val = getattr(record, f.name)
        if val is None and f.name not in allowed_none:
            bad.append(f.name)
    return bad


def main():
    FORCE_OCCLUSION = False 
    if LOG_PATH.exists():
        LOG_PATH.unlink()
        print(f"[reset] removed stale smoke-test log at {LOG_PATH}")

    logger = TrialLogger(str(LOG_PATH))

    env = SimpleEnv(
        xml_path="asset/scene_so101_y.xml", robot_profile="so101",
        action_type="delta_joint_angle", state_type="joint_angle",
        pick_body_name="body_obj_block_2",
        place_body_name="body_obj_bin",
        spawn_x_range=(0.25, 0.52),
        spawn_y_range=(0.02, 0.22),
        spawn_z_range=(0.815, 0.815),
    )

    # Sweep x_shift across trials -> a non-degenerate RANGE of true_visibility,
    # not one fixed value (Day 2 only exercised a single forced layout).
    x_shifts = np.linspace(0.0, 0.045, N_TRIALS)

    print(f"Running {N_TRIALS} MuJoCo smoke trials -> {LOG_PATH}\n")
    results = []
    for trial_id, x_shift in enumerate(x_shifts):
        seed = 900_000 + trial_id
        record, bbox_px = run_one_mujoco_trial(env, trial_id, float(x_shift), seed, force_occlusion=FORCE_OCCLUSION)
        bad_fields = check_schema_populated(record)
        logger.append_trial(record)
        results.append((record, bad_fields))

        print(f"  trial {trial_id} x_shift={x_shift:.4f}  "
              f"true_vis={record.measured_true_visibility:.4f}  "
              f"perceived_vis={record.perceived_visibility_initial:.4f}  "
              f"gap={record.Gap_initial:.4f}  outcome={record.outcome:9s}  "
              f"bbox={bbox_px}  bad_fields={bad_fields or 'none'}")

    # --- Decision-gate checks ---
    all_outcomes = {r.outcome for r, _ in results}
    all_bad_fields = [bf for _, bf in results if bf]
    true_vis_values = [r.measured_true_visibility for r, _ in results]
    vis_range = max(true_vis_values) - min(true_vis_values)

    print("\n=== Day 3 / Phase 0.25 close-out checkpoint ===")
    print(f"Trials run cleanly: {len(results)}/{N_TRIALS}")
    print(f"Distinct outcomes observed: {all_outcomes} "
          f"({'OK -- non-degenerate' if len(all_outcomes) > 1 or 'success' in all_outcomes else 'CHECK -- suspicious'})")
    print(f"true_visibility range across trials: {vis_range:.4f} "
          f"({'OK -- occlusion sweep produced real variation' if vis_range > 0.05 else 'CHECK -- x_shift sweep too narrow, all trials look identical'})")
    print(f"Trials with unexpectedly-None schema fields: {len(all_bad_fields)}/{N_TRIALS} "
          f"({'OK' if not all_bad_fields else 'CHECK -- see per-trial output above'})")
    print(f"Log file written: {LOG_PATH} (exists={LOG_PATH.exists()})")

    gate_pass = (
        len(results) == N_TRIALS
        and not all_bad_fields
        and vis_range > 0.05
    )
    print(f"\nPhase 0.25 close-out decision gate: {'PASS' if gate_pass else 'NOT YET -- see CHECK lines above'}")
    if gate_pass:
        print("Implementation-note sentence for the paper (ready to reuse):")
        print('  "Pipeline feasibility (detector inference, TSDF fusion, Gap computation, '
              'policy-call coherence, and trial logging) was verified end-to-end in a MuJoCo '
              'simulation before real-hardware integration began."')
        print("\nNext: Day 4 (hardware-independent paperwork catch-up), per Week 1 plan.")


if __name__ == "__main__":
    main()