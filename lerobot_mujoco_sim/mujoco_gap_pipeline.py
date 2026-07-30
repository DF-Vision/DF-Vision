"""
mujoco_gap_pipeline.py

Phase 0.25, Days 1-3 (consolidated) -- the real-physics perception module
run_trial.py's USE_REAL_MUJOCO_PHYSICS branch already imports against this
exact contract:

    get_perception_and_ground_truth(env, target_body, occluder_body)
        -> (perceived_visibility: float,
            true_visibility: float,
            perceived_target_xyz: np.ndarray[3])

WHERE THIS CODE CAME FROM (per the Day 1 plan's explicit instruction)
---------------------------------------------------------------------------
world_to_camera(), aabb_world_corners(), project_to_pixels(),
build_bbox_mask(), build_pinhole_extrinsic() below are PORTED VERBATIM from
mujoco_day2.py -- not re-derived. That file's world_to_camera() is the
function that already had the axis-permutation bug and was already fixed
and validated (Day 2's z(depth) sanity check, Day 3's debug-image overlay
confirming green/red bboxes actually land on the real blocks). Copying it
here (rather than importing mujoco_day2 directly) keeps Day 2/3 untouched
as historical/frozen scripts, per the Day 1 plan -- this file is the new
"real" pipeline entry point going forward.

get_object_geometry_and_pose() and render_rgbd() are NOT re-implemented --
they're imported directly from mujoco_render.py, since that file was
already flagged Day 1 as generic/reusable, not something to fork.

Perceived bbox = projected GROUND-TRUTH AABB (Phase 0.25 convention, per
project decision): this is not a placeholder oversight, it's the explicit
Phase 0.25 design -- perceived_visibility here tests the TSDF/rendering
pipeline, not a detector. compute_perceived_visibility() still comes out
of REAL rendered depth (with real occlusion baked into MuJoCo's z-buffer),
so it is not the same number as true_visibility even though the bbox
itself is ground-truth-derived. See RUNNING_NOTES.md's Day-2 entry.

NEW IN THIS FILE (did not exist in Day 2/3): perceived_target_xyz
---------------------------------------------------------------------------
Day 2/3 only ever computed visibility SCORES -- neither script needed a
graspable 3D point. run_trial.py's real-physics branch does need one, and
the CRITICAL correctness rule (stated in run_trial.py's own docstring, and
re-stated in the handoff doc) is that this point must come from
bbox-detection + depth-reprojection, NEVER from
mujoco_render.get_object_geometry_and_pose()'s world-frame ground truth --
feeding ground truth here would silently defeat the entire Gap-driven-
failure experimental logic, exactly as flagged everywhere else in this
project.

Concretely:
  1. Take the bbox center pixel (u_c, v_c) -- the same bbox already built
     for the TSDF mask, not a separate detection.
  2. Sample REAL rendered depth (not ground truth) at that bbox region,
     take the median of valid (>0) pixels -- median, not mean, so a
     partially-occluding closer object (invalid/zero pixels aside) doesn't
     get silently averaged with the target's own surface in a biased way.
  3. Pinhole-unproject (u_c, v_c, median_depth) into camera-frame xyz.
  4. Transform camera-frame -> world-frame via camera_to_world(), the
     ALGEBRAIC INVERSE of mujoco_day2.world_to_camera() -- derived below,
     not assumed from a generic convention (same discipline that caught
     the original T_view bug).

Edge case: if the bbox region has NO valid depth (e.g. fully occluded, or
projected entirely off-frame), we do NOT fall back to ground truth -- that
would silently reintroduce the exact bug this rule exists to prevent.
Instead this raises PerceptionFailure explicitly, so the caller (run_trial.py)
can log/handle a genuine "perception failed" trial rather than get a
falsely-clean number.

Container assumptions (mujoco-rehearsal-camera, per RUNNING_NOTES.md):
    docker cp mujoco_gap_pipeline.py mujoco-rehearsal-camera:/workspace/
    (mujoco_render.py, tsdf_fusion.py, gap_computer.py, occlusion_estimator.py
     must already be present -- Day 1/2 copies)
    export MUJOCO_GL=egl
    xvfb-run -a python3 mujoco_gap_pipeline.py   # self-test, one trial
"""

from __future__ import annotations

import numpy as np

import mujoco_render
from tsdf_fusion import MaskedTSDFFusion, CameraIntrinsics
from occlusion_estimator import ObjectGeometry, Pose, compute_true_visibility

CAM_NAME = "topview"  # matches Day 2/3's camera choice


class PerceptionFailure(RuntimeError):
    """Raised when the bbox region has no valid rendered depth to
    reproject -- caller must handle this as a genuine failed-perception
    trial, not silently fall back to ground truth.

    Carries perceived_visibility/true_visibility -- BOTH already computed
    by the time reprojection fails, since reprojection is the last step in
    get_perception_and_ground_truth() -- so the caller can still log a
    schema-valid TrialRecord with real Gap numbers, even though no
    perceived_target_xyz exists and no grasp should be attempted. Without
    this, a caller would be forced to either fabricate placeholder
    visibility numbers or silently drop the trial from the log entirely --
    both worse than surfacing the real numbers alongside an explicit
    failure. Both attributes are None only if the failure somehow happened
    before either was computed (not expected given the current pipeline
    order, but not silently assumed impossible)."""

    def __init__(self, message, perceived_visibility=None, true_visibility=None):
        super().__init__(message)
        self.perceived_visibility = perceived_visibility
        self.true_visibility = true_visibility


# ---------------------------------------------------------------------------
# Glue, PORTED VERBATIM from mujoco_day2.py (already validated -- see
# module docstring). Do not "clean up" or re-derive these without re-running
# Day 2's sanity check (target camera-frame z > 0, matches measured depth)
# and Day 3's debug-image overlay check.
# ---------------------------------------------------------------------------

def world_to_camera(points_world, T_view):
    """points_world: (N,3) world-frame points. T_view: 4x4 camera->world.
    Ported from mujoco_day2.py -- see that file for the derivation notes
    (raw[0]=depth/forward, raw[1]~horizontal negated, raw[2]~vertical negated)."""
    R = T_view[:3, :3]
    C = T_view[:3, 3]
    raw = (points_world - C[None, :]) @ R
    M = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])
    return raw @ M  # columns: [pinhole_x, pinhole_y, pinhole_z=depth]


def camera_to_world(points_cam, T_view):
    """
    ALGEBRAIC INVERSE of world_to_camera() above -- new in this file, since
    neither Day 2 nor Day 3 needed to go this direction.

    Forward:  pinhole = ((p_world - C) @ R) @ M
    Therefore, since both R (rotation) and M (orthogonal permutation/sign
    matrix, M @ M.T = I) are orthogonal:
        pinhole @ M.T          = (p_world - C) @ R
        (pinhole @ M.T) @ R.T  =  p_world - C
        p_world = (pinhole @ M.T) @ R.T + C

    points_cam: (N,3) camera-frame (pinhole x, pinhole y, depth) points.
    T_view: 4x4 camera->world transform, SAME T_view used for the forward
    call this is inverting -- do not mix T_view from different frames/trials.
    """
    R = T_view[:3, :3]
    C = T_view[:3, 3]
    M = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])
    raw = points_cam @ M.T
    return raw @ R.T + C[None, :]


def aabb_world_corners(center: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """8 corners of an axis-aligned world-frame box. Ported from mujoco_day2.py."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return center[None, :] + signs * half_extents[None, :]


def project_to_pixels(points_cam: np.ndarray, fx, fy, cx, cy):
    """Pinhole projection, camera-frame (N,3) -> pixel u,v + depth z.
    Ported from mujoco_day2.py."""
    z = points_cam[:, 2]
    u = fx * (points_cam[:, 0] / z) + cx
    v = fy * (points_cam[:, 1] / z) + cy
    return u, v, z


def build_bbox_mask(u, v, width, height):
    """Rectangular 0/1 mask spanning the projected corners' pixel extent,
    clipped to image bounds. Ported from mujoco_day2.py."""
    u0, u1 = int(np.clip(np.floor(u.min()), 0, width)), int(np.clip(np.ceil(u.max()), 0, width))
    v0, v1 = int(np.clip(np.floor(v.min()), 0, height)), int(np.clip(np.ceil(v.max()), 0, height))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[v0:v1, u0:u1] = 1
    return mask, (u0, v0, u1, v1)


def build_pinhole_extrinsic(T_view):
    """World->camera extrinsic in the SAME pinhole convention as
    world_to_camera(), for Open3D's TSDFVolume.integrate(). Ported from
    mujoco_day2.py."""
    R = T_view[:3, :3]
    C = T_view[:3, 3]
    M = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])
    R_ext = M.T @ R.T
    t_ext = -R_ext @ C
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_ext
    extrinsic[:3, 3] = t_ext
    return extrinsic


# ---------------------------------------------------------------------------
# NEW: camera-aware base_xy picker
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS: mujoco_day2.py's _pick_safe_base_xy() picks whichever
# spawn-range corner is FARTHEST from the bin -- a correct fix for Day 3's
# bin-overlap bug, but it never considered camera position. Measured
# directly (2026-07-17 sweep): topview camera sits at world (0.3, 0.0);
# the corner-picker chose (0.52, 0.22), roughly 30 degrees off the camera's
# vertical axis at this rig's ~0.49m camera height above the table. At
# that off-axis angle, force_occlusion_layout's height_delta lift (0.06m)
# produces a large LATERAL pixel shift from pure perspective, not just a
# scale change -- confirmed by hand: target bbox_px=(576,476,609,509),
# occluder bbox_px=(600,500,639,539), only ~7% area overlap despite the
# occluder being placed directly above the target in world XY. This is why
# true_visibility collapsed to ~1.0 for every x_shift >= 0.01: the objects
# were never meaningfully overlapping in image space to begin with.
#
# FIX: don't reimplement/edit mujoco_day2.py (kept frozen/historical per
# project convention). Instead, pick a base_xy here that's close to the
# camera's own (x,y) -- minimizing parallax -- while still respecting the
# SAME bin-clearance rule Day 3 established, then hand that base_xy
# straight to mujoco_day2.force_occlusion_layout()'s existing base_xy=
# parameter (that function already re-validates bin clearance internally
# regardless of where base_xy came from, so this fix is purely additive).

def pick_camera_aware_base_xy(env, cam_name: str = CAM_NAME,
                               bin_outer_half: float = 0.06, margin: float = 0.03):
    """
    Returns (x, y) within env's spawn range, clear of the bin footprint
    (same rule as mujoco_day2._pick_safe_base_xy), chosen to MINIMIZE
    lateral distance to the camera's own (x, y) -- so a pure height offset
    (force_occlusion_layout's height_delta) stays close to a pure scale
    change in image space, instead of dragging sideways via parallax.

    Renders once (rgb/depth discarded) purely to read T_view's camera
    position -- camera pose is static regardless of object placement, so
    this is safe to call before positioning target/occluder.
    """
    _, _, _, T_view = mujoco_render.render_rgbd(env, cam_name)
    camera_xy = T_view[:3, 3][:2]

    x_lo, x_hi = env.spawn_x_range
    y_lo, y_hi = env.spawn_y_range

    # First choice: camera's own (x,y) clipped into the spawn range --
    # the closest any legal placement can get to directly under the camera.
    clipped = (float(np.clip(camera_xy[0], x_lo, x_hi)),
               float(np.clip(camera_xy[1], y_lo, y_hi)))
    corners = [(x_lo, y_lo), (x_lo, y_hi), (x_hi, y_lo), (x_hi, y_hi)]
    candidates = [clipped] + corners

    bin_center, _, _ = mujoco_render.get_object_geometry_and_pose(env, "body_obj_bin")

    def is_clear(pt):
        dx = abs(pt[0] - bin_center[0])
        dy = abs(pt[1] - bin_center[1])
        return not (dx < bin_outer_half + margin and dy < bin_outer_half + margin)

    candidates_by_camera_dist = sorted(
        candidates, key=lambda p: (p[0] - camera_xy[0]) ** 2 + (p[1] - camera_xy[1]) ** 2
    )
    for cand in candidates_by_camera_dist:
        if is_clear(cand):
            print(f"[pick_camera_aware_base_xy] camera_xy={tuple(camera_xy)}, "
                  f"bin_center={bin_center[:2]}, chosen base_xy={cand} "
                  f"(closest-to-camera candidate clear of bin)")
            return cand

    raise RuntimeError(
        f"pick_camera_aware_base_xy: no candidate in spawn_x_range={env.spawn_x_range}, "
        f"spawn_y_range={env.spawn_y_range} is both clear of the bin (center={bin_center[:2]}) "
        f"and available -- spawn range may need widening, or bin_outer_half/margin need review."
    )


# ---------------------------------------------------------------------------
# NEW: bbox + depth -> world-frame perceived point
# ---------------------------------------------------------------------------

def _reproject_bbox_to_world(bbox_px, depth_mm, T_view, fx, fy, cx, cy):
    """
    bbox_px: (u0, v0, u1, v1) pixel-space bbox (same one used for the TSDF mask)
    depth_mm: (H, W) float32 REAL rendered depth (not ground truth), millimeters

    Returns perceived_target_xyz (3,) in WORLD frame, or raises
    PerceptionFailure if no valid depth exists inside the bbox.
    """
    u0, v0, u1, v1 = bbox_px
    depth_region = depth_mm[v0:v1, u0:u1]
    valid = depth_region[depth_region > 0]
    if valid.size == 0:
        raise PerceptionFailure(
            f"No valid rendered depth inside bbox {bbox_px} -- cannot reproject "
            f"a perceived target position. This is a genuine failed-perception "
            f"trial (e.g. target fully occluded or off-frame), not a bug to "
            f"paper over with a ground-truth fallback."
        )
    median_depth_mm = float(np.median(valid))
    z_m = median_depth_mm / 1000.0

    u_c = (u0 + u1) / 2.0
    v_c = (v0 + v1) / 2.0
    x_cam = (u_c - cx) * z_m / fx
    y_cam = (v_c - cy) * z_m / fy
    point_cam = np.array([[x_cam, y_cam, z_m]], dtype=np.float64)

    point_world = camera_to_world(point_cam, T_view)[0]
    return point_world.astype(np.float32)


# ---------------------------------------------------------------------------
# Main contract function -- what run_trial.py actually calls
# ---------------------------------------------------------------------------

def get_perception_and_ground_truth(env, target_body: str, occluder_body: str,
                                     cam_name: str = CAM_NAME):
    """
    See module docstring for the full contract. One real MuJoCo render,
    real TSDF fusion for perceived_visibility, real analytic occlusion
    model for true_visibility, and a bbox+depth-reprojected (NOT
    ground-truth) perceived_target_xyz.

    Raises
    ------
    PerceptionFailure if the bbox region has no valid rendered depth to
    reproject (see _reproject_bbox_to_world). Caller must handle this --
    do not catch-and-fall-back-to-ground-truth anywhere upstream.
    """
    rgb, depth_mm, mrender_intr, T_view = mujoco_render.render_rgbd(env, cam_name)
    width, height = mrender_intr.width, mrender_intr.height
    fx, fy, cx, cy = mrender_intr.fx, mrender_intr.fy, mrender_intr.cx, mrender_intr.cy

    target_center_w, _, target_half = mujoco_render.get_object_geometry_and_pose(env, target_body)
    occ_center_w, _, occ_half = mujoco_render.get_object_geometry_and_pose(env, occluder_body)

    target_center_cam = world_to_camera(target_center_w[None, :], T_view)[0]
    occ_center_cam = world_to_camera(occ_center_w[None, :], T_view)[0]
    assert target_center_cam[2] > 0, (
        f"Target camera-frame z={target_center_cam[2]:.4f} not positive -- "
        f"T_view convention check failed (same check as Day 2/3)."
    )

    # --- Perceived bbox: projected ground-truth AABB (Phase 0.25 convention,
    # per project decision -- NOT a real detector yet, see module docstring) ---
    corners_w = aabb_world_corners(target_center_w, target_half)
    corners_cam = world_to_camera(corners_w, T_view)
    u, v, z = project_to_pixels(corners_cam, fx, fy, cx, cy)
    assert np.all(z > 0), "Target has a projected corner behind the camera."
    mask, bbox_px = build_bbox_mask(u, v, width, height)

    # --- perceived_visibility: real tsdf_fusion.py on real rendered depth ---
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

    # --- true_visibility: real occlusion_estimator.py, camera-frame poses ---
    target_pose = Pose(x=target_center_cam[0], y=target_center_cam[1], z=target_center_cam[2])
    target_geom = ObjectGeometry(width=2 * target_half[0], height=2 * target_half[1], depth=2 * target_half[2])
    occ_pose = Pose(x=occ_center_cam[0], y=occ_center_cam[1], z=occ_center_cam[2])
    occ_geom = ObjectGeometry(width=2 * occ_half[0], height=2 * occ_half[1], depth=2 * occ_half[2])
    camera_intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    true_visibility = compute_true_visibility(target_pose, target_geom, occ_pose, occ_geom, camera_intrinsics)

    # --- perceived_target_xyz: bbox + REAL depth reprojection, NEVER ground truth ---
    try:
        perceived_target_xyz = _reproject_bbox_to_world(bbox_px, depth_mm, T_view, fx, fy, cx, cy)
        
        # DEBUG ONLY (do not use this value for control)
        gt_xyz = env.get_obj_pose()[0]

        print("\n========== TARGET COMPARISON ==========")
        print("Perceived target :", perceived_target_xyz)
        print("Ground truth     :", gt_xyz)
        print("Difference (m)   :", perceived_target_xyz - gt_xyz)
        print("=======================================\n")

    except PerceptionFailure as e:
        # Re-raise WITH the visibility numbers attached -- see PerceptionFailure's
        # docstring for why this matters to the caller.
        raise PerceptionFailure(str(e), perceived_visibility=perceived_visibility,
                                 true_visibility=true_visibility) from e

    return perceived_visibility, true_visibility, perceived_target_xyz


# ---------------------------------------------------------------------------
# Self-test: one trial, real numbers, eyeballed against ground truth --
# same discipline as every other file in this project (verify before scaling).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from mujoco_env.y_env import SimpleEnv
    from mujoco_day2 import force_occlusion_layout  # reuse Day 2's known-good pinned layout

    env = SimpleEnv(
        xml_path="asset/scene_so101_y.xml", robot_profile="so101",
        action_type="eef_pose", state_type="joint_angle",
        pick_body_name="body_obj_block_1",    # FIXED 2026-07-17: was block_2 (occluder),
                                              # must match target_body below
        place_body_name="body_obj_bin",
        spawn_x_range=(0.25, 0.52),
        spawn_y_range=(0.02, 0.22),
        spawn_z_range=(0.815, 0.815),
    )

    # Camera-aware base_xy (see pick_camera_aware_base_xy's docstring/comment
    # block above -- fixes the parallax bug found in the 2026-07-17 x_shift
    # sweep, where mujoco_day2's default corner-picker chose a spawn point
    # ~30deg off the camera's axis and occlusion never exceeded ~7% overlap
    # regardless of x_shift).
    base_xy = pick_camera_aware_base_xy(env)

    print("\n=== x_shift sweep (confirms true_visibility actually varies) ===")
    for x_shift in [0.0, 0.01, 0.02, 0.03, 0.04]:
        force_occlusion_layout(env, base_xy=base_xy, x_shift=x_shift)
        pv, tv, xyz = get_perception_and_ground_truth(
            env, target_body="body_obj_block_1", occluder_body="body_obj_block_2"
        )
        print(f"x_shift={x_shift:.3f}  perceived_vis={pv:.4f}  true_vis={tv:.4f}  gap={abs(pv - tv):.4f}")

    # Single-trial detail view at x_shift=0.0 (max expected overlap)
    force_occlusion_layout(env, base_xy=base_xy, x_shift=0.0)
    perceived_vis, true_vis, perceived_xyz = get_perception_and_ground_truth(
        env, target_body="body_obj_block_1", occluder_body="body_obj_block_2"
    )
    ground_truth_xyz, _, _ = mujoco_render.get_object_geometry_and_pose(env, "body_obj_block_1")

    print("\n=== mujoco_gap_pipeline.py self-test (x_shift=0.0 detail) ===")
    print(f"perceived_visibility   = {perceived_vis:.4f}")
    print(f"true_visibility        = {true_vis:.4f}")
    print(f"perceived_target_xyz   = {perceived_xyz}  (bbox+depth reprojection)")
    print(f"ground_truth_xyz       = {ground_truth_xyz}  (env.get_obj_pose-equivalent -- "
          f"NOT what was fed to any grasp, shown only for comparison)")
    print(f"position error (perceived vs ground truth) = "
          f"{np.linalg.norm(perceived_xyz - ground_truth_xyz):.4f} m")
    print("\nSanity checks:")
    print(f"  perceived_visibility in [0,1]: {0.0 <= perceived_vis <= 1.0}")
    print(f"  true_visibility in [0,1]:      {0.0 <= true_vis <= 1.0}")
    print(f"  perceived_xyz finite:          {np.all(np.isfinite(perceived_xyz))}")
    print(f"  position error < 0.15m (sane): {np.linalg.norm(perceived_xyz - ground_truth_xyz) < 0.15}")