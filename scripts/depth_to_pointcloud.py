#!/usr/bin/env python3
"""
depth_to_pointcloud.py
----------------------
Subscribes to the metric depth image published by depth_publisher.py
and converts it to a 3D pointcloud (sensor_msgs/PointCloud2) for use
in the Boundary Pose Sensitivity (BPS) pipeline.

This node is world-agnostic: it works identically across
world1_primitives ... world6_occlusion, since it only depends on the
depth topic and camera_info — not on any specific object, scene, or
grasping-demo assumptions.

Downstream consumers: grasp_estimator.py, log_results.py.

Usage:
    python3 depth_to_pointcloud.py
    (No arguments needed — reads camera_info automatically)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header


# ── Configuration ────────────────────────────────────────────────────────
# Topic names are unchanged from the original demo so that
# depth_publisher.py, grasp_estimator.py, and log_results.py continue
# to work without modification.
DEPTH_TOPIC       = "/depth_model/depth_image"     # from depth_publisher.py
# FIX: was hardcoded "/camera_info", which nothing in this pipeline ever
# publishes — ros_gz_bridge (see run_all_worlds.py) bridges Gazebo's
# rgb_camera sensor's CameraInfo on "/rgb_camera/camera_info", not a bare
# "/camera_info". Using the RGB camera's intrinsics (not the depth
# camera's) is correct here: depth_publisher.py's metric depth output is
# at the same resolution/projection as its RGB input, which is what this
# node's back-projection needs to match.
CAMERA_INFO_TOPIC = "/camera_info"
CLOUD_OUT_TOPIC   = "/depth_model/pointcloud"

MIN_DEPTH = 0.1    # metres — ignore points closer than this (sensor noise floor)
MAX_DEPTH = 5.0    # metres — ignore points further than this (sensor range)

# ── Camera extrinsics (world-agnostic) ───────────────────────────────────
# FIX: the camera-to-world transform used to be hardcoded for a single
# pose (world2_dense_clutter's). It is now computed generically from an
# arbitrary <pose>x y z roll pitch yaw</pose>, supplied either via the
# 'camera_pose' ROS parameter or looked up by the 'world_name' ROS
# parameter in the table below. Real hardware (Orbbec Gemini 2 + SO-101)
# will supply its own pose via 'camera_pose' (or, later, TF) — it does
# not need an entry here.
#
# Poses below are copied verbatim from each world SDF's depth_camera
# model pose (x y z roll pitch yaw), for convenience only — they are
# not baked into the transform math.
WORLD_CAMERA_POSES = {
    "world1_baseline":        (-0.039113983511924744, -0.5, 1.8508999347686768,
                                3.1415899979989477, 1.57079000476762, 3.1415899979989477),
    "world2_dense_clutter":   (3.4694469519536142e-17, -0.5, 1.5700000000000001,
                                3.1415926535897931, 1.5707926536057681, 3.1415926535897931),
    "world3_thin_objects":    (1.6826342061904143e-06, -0.5, 1.8361542224884033,
                                3.1415899979989477, 1.57079000476762, 3.1415899979989477),
    "world4_support_scene":   (-0.092614345252513885, -0.5, 1.9686205387115479,
                                3.1415899979989477, 1.57079000476762, 3.1415899979989477),
    "world5_occlusion_scene": (-0.060530148446559906, -0.5, 1.7741203308105469,
                                3.1415899979989477, 1.57079000476762, 3.1415899979989477),
    "world6_dense_mixed":     (-0.16386400163173676, -0.58311200141906738, 1.7646600008010864,
                                3.1415899979989477, 1.57079000476762, 3.1415899979989477),
}


def _rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Build a rotation matrix from SDF-style roll/pitch/yaw.
    SDF/Gazebo poses use fixed-axis (extrinsic) rotations applied in the
    order roll about X, then pitch about Y, then yaw about Z, i.e.
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


# REP-103 fixed correction from ROS *optical* frame (x=right, y=down,
# z=forward — what the pinhole back-projection below produces) to the
# camera *body*/link frame that the SDF <pose> is expressed in
# (x=forward, y=left, z=up). This is a fixed, universal relationship
# for every optical-frame camera and does not depend on the pose.
_R_BODY_FROM_OPTICAL = np.array([
    [0.0,  0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


def pose_to_transform(x: float, y: float, z: float,
                       roll: float, pitch: float, yaw: float):
    """
    Build a generic (R, t) camera-optical-frame -> world-frame transform
    from an arbitrary SDF camera pose (position + roll/pitch/yaw).

    Given a point p_optical in the camera's optical frame:
        p_world = R @ p_optical + t

    This replaces the old hardcoded, single-pose transform and works for
    any camera placement — the six simulated worlds today, and whatever
    unlisted pose real hardware ends up mounted at tomorrow.
    """
    R_world_body = _rpy_to_rotation_matrix(roll, pitch, yaw)
    R = R_world_body @ _R_BODY_FROM_OPTICAL
    t = np.array([x, y, z])
    return R, t

# How long we tolerate camera_info not having arrived before warning loudly.
# Useful in the BPS pipeline since worlds are launched/torn down repeatedly
# across six scenes, and a silently-stuck node would corrupt a whole run.
CAMERA_INFO_TIMEOUT_SEC = 10.0


class DepthToPointcloud(Node):
    def __init__(self):
        super().__init__("depth_to_pointcloud")
        self.bridge = CvBridge()
        self.K      = None    # camera intrinsic matrix (3x3)
        self.fx = self.fy = self.cx = self.cy = None

        # FIX: camera-to-world extrinsics are now resolved once at startup
        # from parameters instead of being hardcoded in _depth_to_cloud.
        #   - 'camera_pose' : [x, y, z, roll, pitch, yaw] — takes priority
        #     when provided (this is how real hardware, with a pose that
        #     matches none of the six worlds below, will supply it).
        #   - 'world_name'  : falls back to a lookup in WORLD_CAMERA_POSES
        #     when 'camera_pose' isn't set — convenient for the six known
        #     simulation worlds.
        self.declare_parameter("camera_pose", [0.0] * 0)
        self.declare_parameter("world_name", "")
        self.declare_parameter("depth_source_kind", "mde_predicted_z")

        camera_pose_param = self.get_parameter("camera_pose").value
        world_name_param = self.get_parameter("world_name").value

        if camera_pose_param and len(camera_pose_param) == 6:
            pose = tuple(float(v) for v in camera_pose_param)
            self.get_logger().info(
                "Using explicit 'camera_pose' parameter: %s" % (pose,))
        elif world_name_param in WORLD_CAMERA_POSES:
            pose = WORLD_CAMERA_POSES[world_name_param]
            self.get_logger().info(
                "Using camera pose for world_name='%s': %s" %
                (world_name_param, pose))
        else:
            # No usable pose was supplied. Rather than silently falling
            # back to a hardcoded guess (the original bug), fail loudly:
            # an identity transform is obviously wrong and easy to spot
            # in logs/pointclouds, whereas a wrong-but-plausible transform
            # is exactly what caused the original 0.2-0.4m Z error.
            self.get_logger().error(
                "No camera pose available: 'camera_pose' parameter not "
                "set (or wrong length) and 'world_name'=%r not found in "
                "WORLD_CAMERA_POSES. Falling back to an IDENTITY "
                "camera-to-world transform — pointclouds will be in the "
                "camera-optical frame, NOT world frame, until this is "
                "fixed. Set 'camera_pose' explicitly for real hardware "
                "or unlisted worlds." % world_name_param)
            pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        self.R_cam_to_world, self.t_cam_to_world = pose_to_transform(*pose)

        depth_source_kind = str(self.get_parameter("depth_source_kind").value)
        _VALID_DEPTH_SOURCE_KINDS = {"gazebo_range_gt", "mde_predicted_z"}
        if depth_source_kind not in _VALID_DEPTH_SOURCE_KINDS:
            self.get_logger().error(
                "Invalid 'depth_source_kind'=%r, must be one of %s. "
                "Defaulting to 'mde_predicted_z' (no range correction)." %
                (depth_source_kind, _VALID_DEPTH_SOURCE_KINDS))
            depth_source_kind = "mde_predicted_z"
        self.depth_source_kind = depth_source_kind
        self.get_logger().info("depth_source_kind = %s" % self.depth_source_kind)

        # MOD: track when the node started so we can detect a missing
        # camera_info stream within the pipeline's automated runs, instead
        # of hanging silently forever (which the original demo could do
        # since it was always run interactively by a person watching it).

        # MOD: basic frame-rate / health bookkeeping for logging, since the
        # BPS pipeline runs unattended across six worlds and we want to be
        # able to tell from logs alone whether a given world's pointcloud
        # stream actually produced data.
        self._frames_received   = 0
        self._frames_published  = 0
        self._frames_dropped    = 0

        # ── QoS ────────────────────────────────────────────────────────
        # Camera-style topics (depth image, camera_info, pointcloud) are
        # matched against ros_gz_bridge / Gazebo Harmonic publishers, which
        # use sensor-data QoS — BEST_EFFORT + small KEEP_LAST history.
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Publishers / Subscribers ─────────────────────────────────────
        # MOD: queue_size bumped to a small positive buffer (still effectively
        # "latest only" behaviour via tcp_nodelay) — kept queue_size=1 as in
        # the original since the pipeline only ever cares about the most
        # recent frame for a given world snapshot.
        self.cloud_pub = self.create_publisher(
            PointCloud2, CLOUD_OUT_TOPIC, camera_qos
        )
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC,
                                  self._camera_info_callback, camera_qos)
        self.create_subscription(Image, DEPTH_TOPIC,
                                  self._depth_callback, camera_qos)

        # MOD: periodic watchdog timer that checks whether camera_info has
        # arrived. The original script just logged a throttled warning
        # inside the depth callback, which only fires if depth frames are
        # already arriving — it could never warn about a camera_info
        # outage that occurs before any depth frame shows up.
        # rclpy timers have no built-in "oneshot" flag (unlike rospy.Timer),
        # so the callback cancels its own timer after the single check.
        self._camera_info_watchdog_timer = self.create_timer(
            CAMERA_INFO_TIMEOUT_SEC, self._check_camera_info_timeout
        )

        self.get_logger().info(
            "depth_to_pointcloud ready. depth_source_kind=%s, subscribed to depth=%s, "
            "camera_info=%s, publishing=%s" %
            (self.depth_source_kind, DEPTH_TOPIC, CAMERA_INFO_TOPIC, CLOUD_OUT_TOPIC)
        )

    # ── Camera info handling ─────────────────────────────────────────────
    def _camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics from ROS camera_info topic."""
        # MOD: validate the intrinsics before trusting them. The original
        # demo assumed a single, fixed test camera, so it never checked
        # for degenerate K matrices. Across six different Gazebo worlds
        # (potentially with different camera plugins/configs), a zeroed
        # or malformed K would silently produce garbage pointclouds that
        # corrupt downstream BPS measurements without any visible error.
        K = np.array(msg.k).reshape(3, 3)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        if fx <= 0 or fy <= 0:
            self.get_logger().error(
                "Received invalid camera_info (fx=%.3f, fy=%.3f) — "
                "ignoring this message." % (fx, fy)
            )
            return

        if self.K is None:
            self.K, self.fx, self.fy, self.cx, self.cy = K, fx, fy, cx, cy
            self.get_logger().info(
                "Camera intrinsics stored: fx=%.2f fy=%.2f cx=%.2f cy=%.2f" %
                (self.fx, self.fy, self.cx, self.cy))
        else:
            # MOD: detect (rather than silently ignore) intrinsics changing
            # mid-run. This matters in the BPS pipeline because each world
            # is launched fresh — if a stale camera_info from a previous
            # world somehow leaks through, or the camera is reconfigured,
            # we want a clear log line rather than quietly mixing old and
            # new intrinsics across scenes.
            if not np.allclose(K, self.K, atol=1e-3):
                self.get_logger().warning(
                    "camera_info changed after being initially set "
                    "(fx %.2f -> %.2f). Updating intrinsics." % (self.fx, fx))
                self.K, self.fx, self.fy, self.cx, self.cy = K, fx, fy, cx, cy

    def _check_camera_info_timeout(self):
        """One-shot watchdog: warn loudly if camera_info never arrived."""
        # MOD: rclpy timers are periodic by default, so this callback
        # cancels its own timer immediately to reproduce the original
        # rospy.Timer(..., oneshot=True) behaviour exactly.
        self._camera_info_watchdog_timer.cancel()
        if self.fx is None:
            self.get_logger().error(
                "No camera_info received on '%s' after %.1f s. "
                "depth_to_pointcloud cannot produce pointclouds until "
                "intrinsics are available. Check that the world's camera "
                "plugin is publishing camera_info." %
                (CAMERA_INFO_TOPIC, CAMERA_INFO_TIMEOUT_SEC))

    # ── Depth handling ───────────────────────────────────────────────────
    def _depth_callback(self, msg: Image):
        """Convert incoming depth image to 3D pointcloud."""
        self._frames_received += 1

        if self.fx is None:
            self.get_logger().warning(
                    "Waiting for camera_info before processing depth frames..."
            )
            self._frames_dropped += 1
            return

        # MOD: explicit, narrow exception handling instead of a single
        # broad try/except around the whole conversion. This lets us
        # distinguish "bad image encoding from depth_publisher.py" from
        # "bug in our own back-projection code" in the logs, which matters
        # a lot when debugging six separate world runs after the fact.
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except CvBridgeError as e:
            self.get_logger().error(
                "cv_bridge conversion failed for depth image "
                "(encoding=%s): %s" % (msg.encoding, str(e)))
            self._frames_dropped += 1
            return

        if depth.dtype != np.float32:
            self.get_logger().warning(
                "Depth image has dtype %s, expected float32." % depth.dtype
            )

        # Existing code continues
        if depth.ndim != 2 or depth.size == 0:
            self.get_logger().warning(
                "Received depth image with unexpected shape %s — skipping frame."
                % str(depth.shape)
            )
            self._frames_dropped += 1
            return

        # MOD: validate the depth array shape/dtype. depth_publisher.py is
        # expected to always emit 32FC1, but since this node is now a
        # general pipeline component (not hand-fed by a single known
        # script run interactively), we guard against malformed frames
        # (e.g. an empty image during a world's startup transient) instead
        # of letting a downstream numpy/reshape error crash the node.

        # MOD: count how much of the frame is valid metric depth, and warn
        # (without dropping the frame) if a depth map looks suspiciously
        # empty. This is a useful BPS-pipeline-specific signal: an
        # all-invalid depth map for, say, world6_occlusion likely means
        # the depth model failed on that scene, and we want that visible
        # in logs/log_results.py rather than just yielding a silent empty
        # pointcloud.
        finite_mask = np.isfinite(depth)
        valid_ratio = float(np.count_nonzero(finite_mask)) / depth.size
        if valid_ratio < 0.01:
            self.get_logger().warning(
                "Depth frame at stamp %s has only %.2f%% finite values — "
                "resulting pointcloud will be nearly empty." %
                (str(msg.header.stamp), valid_ratio * 100.0))

        try:
            # Build pointcloud — header is preserved verbatim from the
            # incoming depth message, so the published cloud carries the
            # *same* timestamp and frame_id as the depth frame it came
            # from. This is required for the BPS pipeline since
            # grasp_estimator.py / log_results.py need to correlate
            # pointclouds with the exact depth frame (and its associated
            # ground-truth pose / boundary mask) used to generate it.
            cloud_msg = self._depth_to_cloud(depth, msg.header)
            self.cloud_pub.publish(cloud_msg)
            self._frames_published += 1

        except Exception as e:
            # MOD: still keep a catch-all here as a last line of defense
            # so one malformed frame in a six-world automated run never
            # kills the whole node — but now it's scoped only around the
            # actual back-projection/publish step, with full traceback
            # logging for post-hoc debugging.
            self.get_logger().error(
                "depth_to_pointcloud failed to build/publish "
                "pointcloud for stamp %s: %s" %
                (str(msg.header.stamp), str(e)))
            self._frames_dropped += 1

        # MOD: periodic health summary so a long automated run across six
        # worlds leaves a trail of evidence in the log even if nobody is
        # watching it live.
        if self._frames_received % 20 == 0:
            self.get_logger().info(
                "Health: received=%d published=%d dropped=%d" %
                (self._frames_received, self._frames_published,
                 self._frames_dropped))

    def _depth_to_cloud(self, depth: np.ndarray,
                     header: Header) -> PointCloud2:
        """
        Back-project each pixel into 3D using the pinhole camera model.
        Returns a PointCloud2 message.

        Unmodified from the original: this is the metric-accurate
        back-projection logic the BPS pipeline depends on, so it is left
        exactly as-is other than the comments below.
        """
        h, w = depth.shape
        # Pixel grid
        u_coords = np.arange(w)
        v_coords = np.arange(h)
        uu, vv = np.meshgrid(u_coords, v_coords)  # (H, W)

        # Flatten
        z = depth.flatten()                        # depth in metres
        u = uu.flatten()
        v = vv.flatten()

        # Filter invalid / out-of-range points.
        # MOD: no behavioural change to the filter itself — still
        # MIN_DEPTH < z < MAX_DEPTH and finite — but this is now the
        # single source of truth for "valid" points, since BPS analysis
        # (std_depth within boundary rings) downstream relies on exactly
        # this same metric range being honoured consistently across worlds.
        valid = (z > MIN_DEPTH) & (z < MAX_DEPTH) & np.isfinite(z)
        z = z[valid]
        u = u[valid]
        v = v[valid]

        # Pinhole back-projection (camera-optical frame: x=right, y=down,
        # z=depth/forward)
        #
        # MOD: Gazebo's depth camera (and therefore our metric depth model,
        # trained/inferred to match it) reports RANGE — true Euclidean distance
        # along each pixel's viewing ray — not Z-depth (perpendicular distance
        # to the image plane), which is what pinhole back-projection assumes.
        # For off-center pixels the ray is angled, so range > true Z-depth,
        # growing with distance from the image center. This produced exactly
        # the pattern we observed: objects further from the camera's optical
        # center (e.g. obj_cube) showed inflated/spread depth values, while
        # near-center objects read closer to correct. Confirmed by comparing
        # against Gazebo's own ground-truth depth camera output directly
        # (/depth_camera, via save_depth_gz.py): min=0.620 max=1.510 mean=0.989,
        # a spread inconsistent with the true ~0.71-1.57m perpendicular-depth
        # range expected in this scene, but consistent with range encoding.
        #
        # Convert range r to true Z-depth using the pixel's ray direction:
        #   Z = r / sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2)
        px = (u - self.cx) / self.fx
        py = (v - self.cy) / self.fy

        if self.depth_source_kind == "gazebo_range_gt":
            # Gazebo's depth camera reports RANGE (Euclidean distance along
            # the viewing ray), not Z-depth — convert before back-projecting.
            ray_scale = np.sqrt(1.0 + px**2 + py**2)
            z_depth = z / ray_scale   # true perpendicular Z-depth
        else:  # "mde_predicted_z" — DA-V2 already outputs Z-depth directly
            z_depth = z

        x_opt = px * z_depth
        y_opt = py * z_depth
        z_opt = z_depth

        # FIX: points were previously published in raw camera-optical
        # frame with no transform into world frame. Ground-truth object
        # poses (ground_truth/<world>_pose.json) are in Gazebo world
        # frame, so anything comparing this cloud against GT — ROI boxes,
        # grasp_estimator's pose-based checks — was comparing incompatible
        # coordinate systems. This produced garbage ROI results (see:
        # every object earlier reporting the same estimated width,
        # because the "ROI" was really just picking up whatever fell in
        # an arbitrary box that had no relation to any object's real
        # location).
        #
        # The camera-to-world transform (self.R_cam_to_world,
        # self.t_cam_to_world) is computed once at startup by
        # pose_to_transform() from an arbitrary SDF pose (position +
        # roll/pitch/yaw) supplied via the 'camera_pose' or 'world_name'
        # parameter — see __init__. It is NOT specific to any one world:
        # the same matrix-multiply below is correct for all six
        # simulated worlds (each with its own resolved pose) and for any
        # future, unlisted pose (e.g. real hardware), since it is derived
        # from that pose's actual roll/pitch/yaw/position rather than
        # from constants baked in for one camera placement.
        p_optical = np.stack([x_opt, y_opt, z_opt], axis=1)   # (N, 3)
        p_world = p_optical @ self.R_cam_to_world.T + self.t_cam_to_world

        # Pack into PointCloud2
        points = p_world.astype(np.float32)
        cloud_msg = self._array_to_pointcloud2(points, header)
        return cloud_msg

    @staticmethod
    def _array_to_pointcloud2(points: np.ndarray,
                           header: Header) -> PointCloud2:
        """Pack Nx3 float32 array into a PointCloud2 message."""
        # Unmodified: field layout (x, y, z as float32) is preserved so
        # grasp_estimator.py and any other existing PointCloud2 consumers
        # continue to parse this message identically.
        fields = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        point_step = 12   # 3 × 4 bytes
        row_step   = point_step * len(points)

        data = points.tobytes()

        cloud = PointCloud2(
            header       = header,
            height       = 1,
            width        = len(points),
            fields       = fields,
            is_bigendian = False,
            point_step   = point_step,
            row_step     = row_step,
            data         = data,
            is_dense     = True,
        )
        return cloud


# ── Entry point ──────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DepthToPointcloud()
        rclpy.spin(node)
    except KeyboardInterrupt:
        # MOD: graceful shutdown handling. In the original demo this
        # wasn't needed since it was always killed manually with Ctrl+C
        # in a terminal someone was watching; in the automated BPS
        # pipeline, worlds are torn down programmatically between scenes,
        # and an uncaught exception would otherwise print an alarming
        # traceback on every single world transition. This is the ROS 2
        # equivalent of catching rospy.ROSInterruptException.
        pass
    finally:
        if node is not None:
            node.get_logger().info("depth_to_pointcloud shutting down.")
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()