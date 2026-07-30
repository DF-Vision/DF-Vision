"""
mujoco_render.py -- Day 1 (Phase 0.25): MuJoCo -> RGB-D render helper.

Wraps the ALREADY-EXISTING get_fixed_cam_rgbd_pcd() (mujoco_env/mujoco_parser.py,
~line 2176) and reshapes/converts its output to the EXACT array shapes/dtypes/units
your real tsdf_fusion.py pipeline expects (its CameraIntrinsics dataclass, depth in
millimeters as float32, RGB as uint8 HxWx3). No new rendering code is introduced --
MuJoCo already renders full RGB-D via get_fixed_cam_rgbd_pcd(); this file is purely
the "zero-dependency swap point" adapter, same role tsdf_fusion.py's own docstring
describes for real-camera adapters.

Scope note (per Week 1 Day 1 plan): this file ONLY renders and reshapes. It does NOT
compute Gap, perceived_visibility, or true_visibility -- that begins Day 2, once this
file's output has been confirmed shape/unit-compatible with tsdf_fusion.py. Object
pose is reported here in WORLD frame (direct from MuJoCo state, no marker detection
needed in sim) -- converting to CAMERA frame for occlusion_estimator.py's Pose is
explicitly deferred to Day 2, not done here.

Run inside the mujoco-rehearsal container:
    cd /workspace
    python3 mujoco_render.py --camera topview --object body_obj_block_1
    python3 mujoco_render.py --camera sideview --object body_obj_block_2
"""

import argparse
import sys
import os
from dataclasses import dataclass

import numpy as np
import cv2
import mujoco as mj

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mujoco_env.y_env import SimpleEnv


@dataclass
class MujocoCameraIntrinsics:
    """Field-for-field mirror of tsdf_fusion.CameraIntrinsics (width, height, fx,
    fy, cx, cy) so this can be passed straight into that class's constructor
    without renaming anything -- see tsdf_fusion.py's CameraIntrinsics dataclass
    and its .to_o3d() method."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def get_camera_intrinsics(env: SimpleEnv, cam_name: str, img_height: int, img_width: int) -> MujocoCameraIntrinsics:
    """
    Reconstructs the SAME focal-length formula get_fixed_cam_rgbd_pcd() already
    uses internally for its own point cloud (mujoco_parser.py, ~line 2210):
        focal_scaling = 0.5 * img_height / tan(cam_fov * pi / 360)
        cx, cy = img_width/2, img_height/2
    computed here at FULL resolution (not the downscaled point-cloud resolution
    that method uses internally), since tsdf_fusion.py's MaskedTSDFFusion needs
    full-res intrinsics matching the full-res color/depth frames this file returns,
    not the (deliberately coarser) point-cloud-only resolution.
    """
    parser = env.env  # MuJoCoParserClass instance (env.env, per y_env.py's SimpleEnv)
    cam_idx = parser.cam_names.index(cam_name)
    cam_fov = parser.cam_fovs[cam_idx]  # fovy in degrees (70 for both topview/sideview)

    focal_scaling = 0.5 * img_height / np.tan(cam_fov * np.pi / 360)
    return MujocoCameraIntrinsics(
        width=img_width, height=img_height,
        fx=float(focal_scaling), fy=float(focal_scaling),
        cx=img_width / 2.0, cy=img_height / 2.0,
    )


def render_rgbd(env: SimpleEnv, cam_name: str):
    """
    Calls the REAL, already-existing get_fixed_cam_rgbd_pcd() (no new rendering
    logic here) and converts its output to tsdf_fusion.py's expected conventions.

    Unit conversion (the one thing this adapter actually does):
      get_fixed_cam_rgbd_pcd() returns depth in METERS (see its own near/far
      rescale: `depth = near/(1-depth_raw*(1-near/far))`). tsdf_fusion.py's
      integrate_frame() uses depth_scale=1000.0 -- i.e. it expects raw depth
      values in MILLIMETERS, which Open3D's create_from_color_and_depth()
      divides by depth_scale internally to recover meters. So we multiply by
      1000 here, ONCE, at this adapter boundary -- exactly where a real
      depth-camera driver's native mm output would already live. Neither
      gap_computer.py nor tsdf_fusion.py need any changes for this.

    Returns
    -------
    rgb : (H, W, 3) uint8
    depth_mm : (H, W) float32, millimeters
    intrinsics : MujocoCameraIntrinsics
    T_view : (4, 4) np.array, camera-to-world transform (from get_fixed_cam_rgbd_pcd)
    """
    rgb, depth_m, pcd, T_view = env.env.get_fixed_cam_rgbd_pcd(cam_name=cam_name, downscale_pcd=0.1)
    depth_mm = (depth_m * 1000.0).astype(np.float32)

    h, w = depth_mm.shape
    intrinsics = get_camera_intrinsics(env, cam_name, img_height=h, img_width=w)

    return rgb, depth_mm, intrinsics, T_view


def get_object_geometry_and_pose(env: SimpleEnv, body_name: str):
    """
    Reads ground-truth object pose + AABB half-extents DIRECTLY from MuJoCo's
    own state -- no marker detection needed in simulation, since MuJoCo already
    knows exact geometry (per Week 1 Day 2 plan: "you have exact ground-truth
    poses and dimensions in the simulator -- this is actually easier here than
    on real hardware").

    Returns
    -------
    p_world : (3,) np.array, body position in WORLD frame
    R_world : (3,3) np.array, body orientation in WORLD frame
    half_extents : (3,) np.array, geom half-extents in the geom's own local frame
        (valid directly as a world-frame AABB half-extent ONLY when the body is
        axis-aligned in world frame, e.g. euler="0 0 0" as in obj_blocks.xml --
        this is checked and flagged below, not silently assumed)
    """
    parser = env.env
    p_world, R_world = parser.get_pR_body(body_name=body_name)

    body_id = mj.mj_name2id(parser.model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in the loaded model. "
                          f"Check the name against the scene XML (e.g. 'body_obj_block_1'..'_4', "
                          f"or 'body_obj_bin').")

    geom_ids = [i for i in range(parser.model.ngeom) if parser.model.geom_bodyid[i] == body_id]
    if not geom_ids:
        raise ValueError(f"No geoms found for body '{body_name}'.")
    if len(geom_ids) > 1:
        print(f"  NOTE: body '{body_name}' has {len(geom_ids)} geoms; using geom_ids[0]'s "
              f"size as the half-extent (e.g. 'body_obj_bin' has 5 wall/floor geoms -- "
              f"pick a specific one deliberately if you target the bin, not this default).")
    half_extents = parser.model.geom_size[geom_ids[0]].copy()

    is_axis_aligned = np.allclose(R_world, np.eye(3), atol=1e-3)
    if not is_axis_aligned:
        print(f"  WARNING: body '{body_name}' is NOT axis-aligned in world frame "
              f"(R != identity) -- geom half-extents are only a valid world-frame "
              f"AABB when axis-aligned. A rotated-box ground truth needs full corner "
              f"projection, not a naive half-extent box. Flagging, not silently assuming.")

    return p_world, R_world, half_extents


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default="topview", choices=["topview", "sideview"])
    parser.add_argument("--object", default="body_obj_block_1")
    parser.add_argument("--xml-path", default="asset/scene_so101_y.xml")
    parser.add_argument("--out-dir", default="mujoco_render_out")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("Day 1: MuJoCo RGB-D render + interface compatibility check")
    print("=" * 70)

    print(f"\nLoading SimpleEnv from {args.xml_path} (robot_profile='so101') ...")
    env = SimpleEnv(xml_path=args.xml_path, robot_profile='so101',
                 action_type='delta_joint_angle', state_type='joint_angle',
                 pick_body_name='body_obj_block_2',
                 place_body_name='body_obj_bin',
                 spawn_x_range=(0.25, 0.52),
                 spawn_y_range=(0.02, 0.22),
                 spawn_z_range=(0.815, 0.815))

    print(f"\nRendering '{args.camera}' camera ...")
    rgb, depth_mm, intrinsics, T_view = render_rgbd(env, args.camera)

    print(f"\n--- Frame shapes/dtypes (compare against tsdf_fusion.py's expectations) ---")
    rgb_ok = (rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3)
    print(f"  rgb:       shape={rgb.shape}  dtype={rgb.dtype}  "
          f"(tsdf_fusion.integrate_frame expects (H,W,3) uint8 -- "
          f"{'MATCH' if rgb_ok else 'MISMATCH, fix before wiring in'})")
    print(f"  depth_mm:  shape={depth_mm.shape}  dtype={depth_mm.dtype}  "
          f"min={depth_mm.min():.1f}  max={depth_mm.max():.1f}  mean={depth_mm.mean():.1f}  (mm)")
    print(f"  intrinsics: {intrinsics}")
    print(f"  T_view (camera->world):\n{T_view}")

    rgb_path = os.path.join(args.out_dir, f"{args.camera}_rgb.png")
    depth_path = os.path.join(args.out_dir, f"{args.camera}_depth_mm.npy")
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(depth_path, depth_mm)
    print(f"\nSaved: {rgb_path}, {depth_path}")

    print(f"\n--- Ground-truth object geometry (direct from MuJoCo state, no marker needed) ---")
    p_world, R_world, half_extents = get_object_geometry_and_pose(env, args.object)
    print(f"  object='{args.object}'")
    print(f"  p_world={p_world}")
    print(f"  half_extents (geom-local)={half_extents}")

    print("\n" + "=" * 70)
    print("Day 1 checkpoint summary:")
    print("  - one MuJoCo frame rendered, saved to disk, shape/dtype-checked against")
    print("    tsdf_fusion.py's expected inputs")
    print("  - known required conversion #1 (HANDLED HERE): depth units. MuJoCo gives")
    print("    METERS -> multiplied by 1000 in this file to match tsdf_fusion.py's")
    print("    depth_scale=1000 (mm) convention. Done ONCE, at this adapter boundary --")
    print("    same place a real depth-camera driver's native mm output would live.")
    print("    gap_computer.py and tsdf_fusion.py themselves need ZERO changes.")
    print("  - known required conversion #2 (NOT DONE HERE, Day 2's job): object pose.")
    print("    MuJoCo gives WORLD frame; occlusion_estimator.py's Pose expects CAMERA")
    print("    frame. World-to-camera transform (via inverse of T_view above) is left")
    print("    for Day 2, once Gap computation actually needs it.")
    print("=" * 70)


if __name__ == "__main__":
    main()