import cv2
import numpy as np
from mujoco_env.y_env import SimpleEnv
import mujoco_render
from mujoco_day2 import world_to_camera, force_occlusion_layout

env = SimpleEnv(
    xml_path="asset/scene_so101_y.xml", robot_profile="so101",
    action_type="delta_joint_angle", state_type="joint_angle",
    pick_body_name="body_obj_block_2", place_body_name="body_obj_bin",
    spawn_x_range=(0.25, 0.52), spawn_y_range=(0.02, 0.22), spawn_z_range=(0.815, 0.815),
)
force_occlusion_layout(env)  # same pinned layout as mujoco_day3's trial 0


print("\n========== BODY / GEOM MAP ==========")

for i in range(env.model.ngeom):
    geom_name = mujoco.mj_id2name(
        env.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        i,
    )

    body_id = env.model.geom_bodyid[i]

    body_name = mujoco.mj_id2name(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
    )

    print(f"{i:2d} | body={body_name:25s} | geom={geom_name}")

print("=====================================\n")

rgb, depth_mm, intr, T_view = mujoco_render.render_rgbd(env, "topview")
img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)

bodies_to_check = ["body_obj_block_1", "body_obj_block_2", "body_obj_block_3",
                    "body_obj_block_4", "body_obj_bin", "target", "gripper", "base"]

for name in bodies_to_check:
    try:
        p_world, _, _ = mujoco_render.get_object_geometry_and_pose(env, name)
    except Exception as e:
        print(f"{name}: skipped ({e})")
        continue
    p_cam = world_to_camera(p_world[None, :], T_view)[0]
    if p_cam[2] <= 0:
        print(f"{name}: behind camera, skipped")
        continue
    u = int(intr.fx * (p_cam[0] / p_cam[2]) + intr.cx)
    v = int(intr.fy * (p_cam[1] / p_cam[2]) + intr.cy)
    print(f"{name}: world={p_world}, pixel=({u},{v})")
    cv2.circle(img, (u, v), 4, (0, 255, 255), -1)
    cv2.putText(img, name, (u + 6, v), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

cv2.imwrite("labeled_bodies.png", img)
print("saved labeled_bodies.png")