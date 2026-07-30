"""
mujoco_day2.py

Phase 0.25, Day 2 — wire the REAL tsdf_fusion.py, occlusion_estimator.py, and
gap_computer.py onto MuJoCo-rendered frames, with ZERO code changes to any of
those three modules. Only glue/adapter logic lives here, same convention as
mujoco_render.py on Day 1.

Design (per Day 1 handoff + Week 1 plan):
  - Target: body_obj_block_1
  - Synthetic "YOLO" mask: block_1's own projected AABB bbox in camera-pixel
    space (mirrors what a real detector's bbox would give, occluded or not).
    The mask does NOT need to know about occlusion -- MuJoCo's own renderer
    already bakes real occlusion into the depth buffer via its z-buffer, so
    integrating real depth inside this bbox naturally reflects whatever's
    actually visible in front of the camera. No synthetic depth-blackout
    trickery needed (unlike tsdf_fusion.py's own self-test, which fakes an
    occluder because it has no renderer available).
  - Occluder (ground truth only): body_obj_block_2, used purely as the
    second AABB fed into occlusion_estimator.compute_true_visibility(),
    which computes true_visibility analytically (2D rect projection +
    depth-ordering), independent of any rendered image.
  - perceived_visibility: real tsdf_fusion.MaskedTSDFFusion, fed the masked
    real depth frame, using T_view directly as the fusion volume's
    extrinsic (world-frame fusion, per that module's own docstring
    convention: "extrinsic = camera pose in world frame").
  - true_visibility: real occlusion_estimator.compute_true_visibility(),
    which expects poses in CAMERA frame (pinhole z-forward convention) --
    so world-frame object poses from MuJoCo are explicitly transformed via
    inv(T_view) before being passed in. This is the step Day 1 deferred.
  - Gap: real gap_computer.compute_gap_features() on a single-step
    trajectory ([perceived_visibility], [true_visibility]).

Container assumptions (mujoco-rehearsal, no bind mount, per Day 1 notes):
  Copy these four files flat into /workspace before running, so imports
  resolve with no path surgery:
      docker cp mujoco_render.py        mujoco-rehearsal:/workspace/
      docker cp tsdf_fusion.py          mujoco-rehearsal:/workspace/
      docker cp gap_computer.py         mujoco-rehearsal:/workspace/
      docker cp occlusion_estimator.py  mujoco-rehearsal:/workspace/
      docker cp mujoco_day2.py          mujoco-rehearsal:/workspace/
  Then, inside the container, every fresh session:
      export MUJOCO_GL=egl
      cd /workspace
      python3 mujoco_day2.py

Known open question (flagged, not silently assumed): whether T_view's
rotation follows the pinhole "+Z forward" convention that occlusion_estimator.py
and tsdf_fusion.py both assume, or an OpenGL "-Z forward" convention. The
sanity check below asserts the target's camera-frame z comes out positive
and in the same ballpark as Day 1's measured depth (~500mm / 0.5m); if that
assertion fails, T_view's rotation needs a 180 deg flip about X before
inversion -- see world_to_camera() below for where to patch it.
"""

import numpy as np

from mujoco_env.y_env import SimpleEnv
import mujoco_render

from tsdf_fusion import MaskedTSDFFusion, CameraIntrinsics
from gap_computer import compute_gap_features
from occlusion_estimator import ObjectGeometry, Pose, compute_true_visibility


# ---------------------------------------------------------------------------
# Glue helpers (new code, not touching any of the three real modules)
# ---------------------------------------------------------------------------

def world_to_camera(points_world, T_view):
    """
    points_world: (N,3) world-frame points
    T_view: 4x4 camera->world transform (R = T_view[:3,:3], C = T_view[:3,3])

    Derived directly from R's columns + cross-checked against known body
    positions vs. the actual rendered image layout (not assumed from a
    generic OpenGL/CV convention) -- see diagnose_camera_frame.py output.
    raw[0] = depth (forward), raw[1] relates to horizontal (negated),
    raw[2] relates to vertical (negated).
    """
    R = T_view[:3, :3]
    C = T_view[:3, 3]
    raw = (points_world - C[None, :]) @ R
    M = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])
    return raw @ M  # columns: [pinhole_x, pinhole_y, pinhole_z=depth]


def aabb_world_corners(center: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """8 corners of an axis-aligned world-frame box."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return center[None, :] + signs * half_extents[None, :]


def project_to_pixels(points_cam: np.ndarray, fx, fy, cx, cy):
    """Pinhole projection, camera-frame (N,3) -> pixel (N,), (N,), z (N,)."""
    z = points_cam[:, 2]
    u = fx * (points_cam[:, 0] / z) + cx
    v = fy * (points_cam[:, 1] / z) + cy
    return u, v, z


def build_bbox_mask(u, v, width, height) -> np.ndarray:
    """Rectangular 0/1 mask spanning the projected corners' pixel extent,
    clipped to image bounds -- stands in for a real YOLO detection bbox."""
    u0, u1 = int(np.clip(np.floor(u.min()), 0, width)), int(np.clip(np.ceil(u.max()), 0, width))
    v0, v1 = int(np.clip(np.floor(v.min()), 0, height)), int(np.clip(np.ceil(v.max()), 0, height))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[v0:v1, u0:u1] = 1
    return mask, (u0, v0, u1, v1)




def build_pinhole_extrinsic(T_view):
    """
    World->camera extrinsic in the SAME pinhole convention as
    world_to_camera() above, for Open3D's TSDFVolume.integrate().
    """
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

def force_occlusion_layout(env, target_body="body_obj_block_1", occluder_body="body_obj_block_2",
                            base_xy=None, table_z=0.815, height_delta=0.06, x_shift=0.008):
    """Pin target + occluder to a deterministic layout with GUARANTEED partial
    occlusion, instead of depending on SimpleEnv's randomized spawn.

    base_xy now defaults to None -> auto-picked clear of the bin's footprint
    (see _pick_safe_base_xy below), instead of the previously hardcoded
    (0.35, 0.12), which was found (Day 3 debug image) to land the pinned
    target/occluder on or inside the bin itself -- the bin's static walls
    were being masked/integrated instead of the actual blocks, which is why
    perceived_visibility was frozen at 0.5833 across every trial regardless
    of x_shift.

    A hard assertion now guards against this regressing silently: whatever
    base_xy is used (default or caller-supplied), it must be measurably
    clear of the bin's outer footprint before any pose is set.
    """
    if base_xy is None:
        base_xy = _pick_safe_base_xy(env)

    x0, y0 = base_xy
    target_xyz = np.array([x0, y0, table_z], dtype=np.float32)
    occluder_xyz = np.array([x0 + x_shift, y0, table_z + height_delta], dtype=np.float32)

    # --- Guard: refuse to proceed if base_xy is inside/near the bin ---
    bin_center, _, _ = mujoco_render.get_object_geometry_and_pose(env, "body_obj_bin")
    bin_outer_half = 0.06  # obj_bin.xml outer footprint, per y_env.py's own
                            # _get_bin_exclusion_bounds -- matches the real bin,
                            # not the possibly-partial geom_size from
                            # get_object_geometry_and_pose's multi-geom warning.
    margin = 0.03
    dx = abs(target_xyz[0] - bin_center[0])
    dy = abs(target_xyz[1] - bin_center[1])
    assert not (dx < bin_outer_half + margin and dy < bin_outer_half + margin), (
        f"force_occlusion_layout: target_xyz={target_xyz[:2]} is within "
        f"{bin_outer_half + margin}m of the bin center {bin_center[:2]} -- "
        f"this is exactly the bug found in Day 3's debug image (bbox drawn on "
        f"the bin, not a block). Choose a base_xy clearly outside the bin's "
        f"footprint, or let base_xy=None auto-pick one."
    )

    env.env.set_p_base_body(body_name=target_body, p=target_xyz)
    env.env.set_R_base_body(body_name=target_body, R=np.eye(3, 3))
    env.env.set_p_base_body(body_name=occluder_body, p=occluder_xyz)
    env.env.set_R_base_body(body_name=occluder_body, R=np.eye(3, 3))

    env.env.data.qvel[:] = 0.0
    env.env.data.qacc[:] = 0.0
    env.env.forward(increase_tick=False)

    print(f"[force_occlusion] {target_body} pinned to {target_xyz}, "
          f"{occluder_body} pinned to {occluder_xyz} (height_delta={height_delta}, "
          f"x_shift={x_shift}, bin_dist=({dx:.3f},{dy:.3f}))")


def _pick_safe_base_xy(env, table_z=0.815):
    """Pick a spawn point inside spawn_x_range/spawn_y_range that's clearly
    clear of the bin's footprint -- avoids hardcoding a point that might
    silently drift into the bin if the scene XML ever changes. Picks the
    corner of the spawn region farthest from the bin center, which is a
    cheap, deterministic, good-enough heuristic here (not a general solver).
    """
    bin_center, _, _ = mujoco_render.get_object_geometry_and_pose(env, "body_obj_bin")
    x_lo, x_hi = env.spawn_x_range
    y_lo, y_hi = env.spawn_y_range
    candidates = [(x_lo, y_lo), (x_lo, y_hi), (x_hi, y_lo), (x_hi, y_hi)]
    best = max(candidates, key=lambda p: (p[0]-bin_center[0])**2 + (p[1]-bin_center[1])**2)
    print(f"[_pick_safe_base_xy] bin_center={bin_center[:2]}, chosen base_xy={best} "
          f"(farthest spawn-range corner from bin)")
    return best


# ---------------------------------------------------------------------------
# Main trial
# ---------------------------------------------------------------------------

def main():
    # Same working constructor as Day 1 (real spawn ranges from manual_collect_data.py)
    env = SimpleEnv(
        xml_path="asset/scene_so101_y.xml", robot_profile="so101",
        action_type="delta_joint_angle", state_type="joint_angle",
        pick_body_name="body_obj_block_2",
        place_body_name="body_obj_bin",
        spawn_x_range=(0.25, 0.52),
        spawn_y_range=(0.02, 0.22),
        spawn_z_range=(0.815, 0.815),
    )

    # Pin block_1/block_2 to a deterministic, guaranteed-partial-occlusion
    # layout instead of trusting random spawn -- set FORCE_OCCLUSION=False
    # below to fall back to whatever SimpleEnv randomly spawned.
    FORCE_OCCLUSION = True
    if FORCE_OCCLUSION:
        force_occlusion_layout(env)

    cam_name = "topview"
    rgb, depth_mm, mrender_intr, T_view = mujoco_render.render_rgbd(env, cam_name)
    width, height = mrender_intr.width, mrender_intr.height
    fx, fy, cx, cy = mrender_intr.fx, mrender_intr.fy, mrender_intr.cx, mrender_intr.cy

    # --- Ground-truth geometry, straight from MuJoCo state (world frame) ---
    # get_object_geometry_and_pose returns (p_world, R_world, half_extents) --
    # R_world is unused here since it's only checked (with a warning) inside
    # that function for axis-alignment; we don't need it for AABB corners.
    target_center_w, _target_R, target_half = mujoco_render.get_object_geometry_and_pose(env, "body_obj_block_1")
    occ_center_w, _occ_R, occ_half = mujoco_render.get_object_geometry_and_pose(env, "body_obj_block_2")

    # --- Camera-frame sanity check (flags T_view convention issues early) ---
    target_center_cam = world_to_camera(target_center_w[None, :], T_view)[0]
    print(f"[sanity] target camera-frame xyz={target_center_cam}, "
          f"z(depth)={target_center_cam[2]*1000:.1f}mm "
          f"(Day 1 measured depth range was ~304-505mm at this camera pose)")
    assert target_center_cam[2] > 0, (
        f"Target camera-frame z={target_center_cam[2]:.4f} is not positive -- "
        f"unexpected given the diagnosed axis permutation; re-check T_view "
        f"if the scene/camera setup changed."
    )
    print(f"[sanity] target camera-frame z={target_center_cam[2]*1000:.1f}mm "
          f"(Day 1 measured depth range was ~304-505mm at this camera pose)")

    # --- Synthetic detection mask: target's own projected AABB bbox ---
    corners_w = aabb_world_corners(target_center_w, target_half)
    corners_cam = world_to_camera(corners_w, T_view)
    u, v, z = project_to_pixels(corners_cam, fx, fy, cx, cy)
    assert np.all(z > 0), "Target has a projected corner behind the camera -- check T_view convention."
    mask, bbox_px = build_bbox_mask(u, v, width, height)
    print(f"[mask] target bbox in pixels: {bbox_px}, mask sum={mask.sum()}")

    # --- perceived_visibility: real tsdf_fusion.py, unmodified ---
    intr = CameraIntrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    fusion = MaskedTSDFFusion(voxel_length=0.004, sdf_trunc=0.02)
    extrinsic = build_pinhole_extrinsic(T_view)
    integ_result = fusion.integrate_frame(
        color=rgb, depth=depth_mm, mask=mask, intrinsics=intr,
        extrinsic=extrinsic,  # world->camera, permuted pinhole convention -- see build_pinhole_extrinsic()
        depth_scale=1000.0, depth_trunc=3.0,
    )
    print(f"[tsdf] integrate_frame: {integ_result}")
    perceived_visibility = fusion.compute_perceived_visibility(
        target_center=target_center_w, target_half_extents=target_half, grid_resolution=12
    )
    print(f"[tsdf] perceived_visibility={perceived_visibility:.4f}")

    # --- true_visibility: real occlusion_estimator.py, unmodified ---
    # occlusion_estimator expects CAMERA-frame poses (pinhole, z-forward).
    occ_center_cam = world_to_camera(occ_center_w[None, :], T_view)[0]

    target_pose = Pose(x=target_center_cam[0], y=target_center_cam[1], z=target_center_cam[2])
    target_geom = ObjectGeometry(width=2 * target_half[0], height=2 * target_half[1], depth=2 * target_half[2])
    occ_pose = Pose(x=occ_center_cam[0], y=occ_center_cam[1], z=occ_center_cam[2])
    occ_geom = ObjectGeometry(width=2 * occ_half[0], height=2 * occ_half[1], depth=2 * occ_half[2])

    camera_intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    true_visibility = compute_true_visibility(target_pose, target_geom, occ_pose, occ_geom, camera_intrinsics)
    print(f"[occlusion_estimator] true_visibility={true_visibility:.4f}")

    if abs(perceived_visibility - true_visibility) < 1e-6 and true_visibility >= 0.999:
        print("[note] true_visibility==1.0 and Gap≈0 -- block_2 is not currently occluding "
              "block_1 from this camera given the sampled spawn positions. Non-degenerate for "
              "a feasibility check (the code path ran end-to-end), but if you want a real "
              "occlusion case for eyeballing, re-run (spawn is randomized) or pin both bodies' "
              "qpos explicitly before rendering.")

    # --- Gap: real gap_computer.py, unmodified, single-step trajectory ---
    gap_result = compute_gap_features([perceived_visibility], [true_visibility])
    print(f"[gap_computer] {gap_result}")

    print("\n=== Day 2 checkpoint ===")
    print(f"perceived_visibility = {perceived_visibility:.4f}  (tsdf_fusion.py, unmodified)")
    print(f"true_visibility      = {true_visibility:.4f}  (occlusion_estimator.py, unmodified)")
    print(f"gap_initial           = {gap_result.gap_initial:.4f}  (gap_computer.py, unmodified)")
    print("All three real modules ran end-to-end on MuJoCo-rendered data with zero code changes "
          "to tsdf_fusion.py, occlusion_estimator.py, or gap_computer.py themselves.")


if __name__ == "__main__":
    main()