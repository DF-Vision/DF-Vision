"""
diagnose_camera_frame.py -- prints T_view's raw rotation matrix and
UNPERMUTED, UNFLIPPED camera-frame coordinates for several known bodies,
spread across the frame (not just one near-axis point). This is the
"read it, don't guess it" fix: R is a rotation matrix, so its columns
directly tell us which world axis maps to which camera-local axis.
"""
import numpy as np
from mujoco_env.y_env import SimpleEnv
import mujoco_render

env = SimpleEnv(
    xml_path="asset/scene_so101_y.xml", robot_profile="so101",
    action_type="delta_joint_angle", state_type="joint_angle",
    pick_body_name="body_obj_block_2",
    place_body_name="body_obj_bin",
    spawn_x_range=(0.25, 0.52),
    spawn_y_range=(0.02, 0.22),
    spawn_z_range=(0.815, 0.815),
)

rgb, depth_mm, intr, T_view = mujoco_render.render_rgbd(env, "topview")

R = T_view[:3, :3]
C = T_view[:3, 3]

print("=== T_view (camera->world) ===")
print(T_view)
print("\n=== R columns (each column = one camera-local axis, expressed in world coords) ===")
for i in range(3):
    print(f"  R col {i}: {R[:, i]}  (dominant world axis: "
          f"{'XYZ'[np.argmax(np.abs(R[:, i]))]}, sign={np.sign(R[np.argmax(np.abs(R[:, i])), i]):+.0f})")
print(f"\nCamera world position C: {C}")

bodies = ["body_obj_block_1", "body_obj_block_2", "body_obj_block_3",
          "body_obj_block_4", "body_obj_bin", "gripper", "base"]

print("\n=== per-body: world pos, and RAW camera-frame coords (p-C)@R, no permutation/flip ===")
for name in bodies:
    try:
        p_world, _, _ = mujoco_render.get_object_geometry_and_pose(env, name)
    except Exception as e:
        print(f"  {name}: could not read pose ({e})")
        continue
    raw = (p_world - C) @ R
    print(f"  {name:20s} world={np.round(p_world,3)}  raw_cam_frame={np.round(raw,4)}")