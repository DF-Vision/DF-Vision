import numpy as np
from mujoco_env.y_env import SimpleEnv
import mujoco_render
import cv2

env = SimpleEnv(
    xml_path="asset/scene_so101_y.xml", robot_profile="so101",
    action_type="delta_joint_angle", state_type="joint_angle",
    pick_body_name="body_obj_block_2", place_body_name="body_obj_bin",
    spawn_x_range=(0.25, 0.52), spawn_y_range=(0.02, 0.22), spawn_z_range=(0.815, 0.815),
)

# Send block_1 and block_2 far away (below the table, out of camera view)
away = np.array([0.35, 0.12, -5.0], dtype=np.float32)
env.env.set_p_base_body(body_name="body_obj_block_1", p=away)
env.env.set_R_base_body(body_name="body_obj_block_1", R=np.eye(3, 3))
env.env.set_p_base_body(body_name="body_obj_block_2", p=away + [0, 0, 0.1])
env.env.set_R_base_body(body_name="body_obj_block_2", R=np.eye(3, 3))
env.env.data.qvel[:] = 0.0
env.env.data.qacc[:] = 0.0
env.env.forward(increase_tick=False)

rgb, depth_mm, intr, T_view = mujoco_render.render_rgbd(env, "topview")
cv2.imwrite("gray_box_isolation_test.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
print("saved gray_box_isolation_test.png -- if the gray box is GONE, it was block_1/block_2. "
      "If it's STILL THERE, it's something else (shadow, another body, or a rendering artifact).")