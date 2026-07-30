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

bin_center, _, bin_half = mujoco_render.get_object_geometry_and_pose(env, "body_obj_bin")
print(f"bin center (world xy) = {bin_center[:2]}")
print(f"bin half-extent (geom-local, may only cover one wall geom -- see Day1's multi-geom warning) = {bin_half}")

base_xy = np.array([0.35, 0.12])
dist = np.linalg.norm(base_xy - bin_center[:2])
print(f"force_occlusion_layout's base_xy = {base_xy}")
print(f"distance from bin center = {dist:.4f} m")

# obj_bin.xml's actual OUTER footprint half-width is 0.06m (per y_env.py's
# own _get_bin_exclusion_bounds -- more reliable than geom_size[0] alone,
# since get_object_geometry_and_pose() warns it may only grab one of the
# bin's several wall/floor geoms).
bin_outer_half = 0.06
inside = (abs(base_xy[0] - bin_center[0]) < bin_outer_half and
          abs(base_xy[1] - bin_center[1]) < bin_outer_half)
print(f"base_xy inside bin's outer footprint? {inside}")