"""
tsdf_fusion.py

Purpose
-------
Compute perceived_visibility from RGB-D + a YOLO detection mask, via Open3D
TSDF (Truncated Signed Distance Field) volumetric fusion. This is the
"what the robot's own perception thinks it sees" half of the Gap -- as
opposed to occlusion_estimator.py, which is the independent ground truth.

This module also provides the active re-observation / re-fusion step used
by C5 (fusing a second viewpoint to reduce Gap after a C4 gate flag).

Design note: this file takes masked depth frames as plain numpy arrays, not
ROS2 messages or Gazebo topics directly. Real camera/topic adapters convert
their output to this format upstream -- this is the project's
"zero-dependency swap point": this file is identical whether the depth
frame came from a synthetic Gazebo camera or a real RealSense.

Approach
--------
1. Integrate one or more masked RGB-D frames into an Open3D
   ScalableTSDFVolume (masked: only pixels the YOLO bbox/mask covers are
   integrated, everything else is treated as unobserved/no-data).
2. Extract a point cloud / mesh from the volume.
3. perceived_visibility = (voxels marked "observed & non-empty" within the
   target's known bounding volume) / (total voxels within that bounding
   volume) -- a simple, auditable ratio, not a learned estimate.

Edge cases handled
-------------------
- Frame with an empty/all-zero mask (YOLO detected nothing): integration
  is skipped for that frame and a warning is returned rather than silently
  integrating garbage or crashing on an empty point set.
- Depth values of 0 (common sentinel for "no return", e.g. glass/transparent
  surfaces per the hardware doc's known RealSense risk): explicitly masked
  out before integration, not treated as a real depth of 0m.
- Multiple-frame re-fusion (C5's active mitigation step): frames are
  integrated into the SAME running volume, not re-created from scratch --
  this is what lets a second viewpoint actually reduce Gap rather than just
  replacing the first estimate.
"""

import numpy as np
import open3d as o3d
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def to_o3d(self) -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy
        )


class MaskedTSDFFusion:
    def __init__(self, voxel_length: float = 0.004, sdf_trunc: float = 0.02):
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        self._n_frames_integrated = 0

    def integrate_frame(self, color: np.ndarray, depth: np.ndarray,
                         mask: np.ndarray, intrinsics: CameraIntrinsics,
                         extrinsic: Optional[np.ndarray] = None,
                         depth_scale: float = 1000.0, depth_trunc: float = 3.0) -> dict:
        """Integrate one masked RGB-D frame into the running TSDF volume.

        Parameters
        ----------
        color : (H, W, 3) uint8
        depth : (H, W) float32 or uint16, in whatever unit depth_scale converts from
        mask : (H, W) bool or 0/1 -- True/1 where YOLO's bbox/mask says the
            target is; everything else is zeroed out before integration.
        extrinsic : (4,4) camera pose in world frame; identity if None
            (single-camera, camera-frame-is-world-frame case).

        Returns
        -------
        dict with 'integrated': bool, 'note': str -- reports whether this
        frame was actually integrated (False if the mask was empty).
        """
        if mask.sum() == 0:
            return {"integrated": False, "note": "empty mask (no detection) -- frame skipped"}

        mask_bool = mask.astype(bool)
        depth_masked = depth.copy().astype(np.float32)
        # Zero/invalid depth (no sensor return, e.g. glass) must not be
        # treated as "object at 0m" -- mask it out explicitly.
        depth_masked[~mask_bool] = 0.0
        depth_masked[depth <= 0] = 0.0

        if depth_masked.sum() == 0:
            return {"integrated": False,
                    "note": "mask present but all masked depth values are invalid/zero "
                            "(e.g. transparent-surface sensor dropout) -- frame skipped"}

        color_masked = color.copy()
        color_masked[~mask_bool] = 0

        color_o3d = o3d.geometry.Image(np.ascontiguousarray(color_masked))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_masked))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=depth_scale, depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        ext = extrinsic if extrinsic is not None else np.eye(4)
        self.volume.integrate(rgbd, intrinsics.to_o3d(), ext)
        self._n_frames_integrated += 1
        return {"integrated": True, "note": f"frame integrated ({self._n_frames_integrated} total)"}

    def compute_perceived_visibility(self, target_center: np.ndarray,
                                      target_half_extents: np.ndarray,
                                      grid_resolution: int = 12) -> float:
        """perceived_visibility = fraction of a grid of sample points inside
        the target's known bounding box that fall within a non-empty voxel
        of the fused TSDF volume (i.e. "the fusion actually has real
        geometry there").

        Parameters
        ----------
        target_center : (3,) array, target's known center in world/camera frame
        target_half_extents : (3,) array, half-width/height/depth
        grid_resolution : samples per axis (grid_resolution^3 total samples)

        Returns
        -------
        float in [0, 1]
        """
        if self._n_frames_integrated == 0:
            return 0.0  # nothing integrated yet -> nothing perceived

        pcd = self.volume.extract_point_cloud()
        if len(pcd.points) == 0:
            return 0.0  # no valid geometry recovered -> perceived_visibility 0

        points = np.asarray(pcd.points)

        # Build a query grid over the target's known bounding box
        lin = [np.linspace(-h, h, grid_resolution) for h in target_half_extents]
        gx, gy, gz = np.meshgrid(*lin, indexing="ij")
        query_points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3) + target_center

        # For each query point, check if ANY fused point lies within one
        # voxel-length's distance (a simple occupancy check, not a full KD-tree
        # radius search with a physically motivated tolerance -- acceptable
        # for this use since it's a coarse presence/absence ratio, not exact
        # surface reconstruction).
        if len(points) == 0:
            return 0.0

        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        tol = 2 * self.volume.voxel_length if hasattr(self.volume, "voxel_length") else 0.01
        dists, _ = tree.query(query_points, k=1)
        occupied = (dists <= tol).sum()
        return float(occupied / len(query_points))


def make_synthetic_depth_frame(width: int, height: int, intrinsics: CameraIntrinsics,
                                target_center: np.ndarray, target_half_extents: np.ndarray,
                                occluder_uv_box: Optional[tuple] = None,
                                dropout_fraction: float = 0.0,
                                seed: int = 0):
    """Build a synthetic color/depth/mask frame of a box-shaped target,
    optionally with a rectangular occluder region blacked out (simulating
    an occluding panel) and/or random depth dropout (simulating glass).
    Used ONLY for self-testing this module -- never a substitute for real
    or Gazebo-rendered frames in actual data collection.
    """
    rng = np.random.default_rng(seed)
    color = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)

    z = target_center[2]
    hw, hh = target_half_extents[0], target_half_extents[1]
    u_center = intrinsics.fx * (target_center[0] / z) + intrinsics.cx
    v_center = intrinsics.fy * (target_center[1] / z) + intrinsics.cy
    u_half = intrinsics.fx * (hw / z)
    v_half = intrinsics.fy * (hh / z)

    u0, u1 = int(max(0, u_center - u_half)), int(min(width, u_center + u_half))
    v0, v1 = int(max(0, v_center - v_half)), int(min(height, v_center + v_half))

    color[v0:v1, u0:u1] = (200, 200, 200)
    depth[v0:v1, u0:u1] = z * 1000.0  # depth_scale=1000 convention (mm)
    mask[v0:v1, u0:u1] = 1

    if occluder_uv_box is not None:
        ou0, ov0, ou1, ov1 = occluder_uv_box
        ou0, ou1 = max(0, ou0), min(width, ou1)
        ov0, ov1 = max(0, ov0), min(height, ov1)
        mask[ov0:ov1, ou0:ou1] = 0
        depth[ov0:ov1, ou0:ou1] = 0.0

    if dropout_fraction > 0:
        dropout_mask = rng.random((height, width)) < dropout_fraction
        depth[dropout_mask & (mask > 0)] = 0.0

    return color, depth, mask


if __name__ == "__main__":
    intr = CameraIntrinsics(width=640, height=480, fx=500, fy=500, cx=320, cy=240)
    target_center = np.array([0.0, 0.0, 0.5])
    target_half = np.array([0.04, 0.04, 0.04])  # ~8cm apple-ish box

    # Case 1: fully visible target, single frame -> perceived_visibility should be high
    color, depth, mask = make_synthetic_depth_frame(640, 480, intr, target_center, target_half)
    fusion = MaskedTSDFFusion(voxel_length=0.004, sdf_trunc=0.02)
    result = fusion.integrate_frame(color, depth, mask, intr)
    assert result["integrated"], f"Expected successful integration, got: {result}"
    pv = fusion.compute_perceived_visibility(target_center, target_half, grid_resolution=8)
    print(f"Case 1 (fully visible, single frame) OK: integrated={result['integrated']}, "
          f"perceived_visibility={pv:.4f}")
    # NOTE: a single depth frame only observes the target's FRONT SURFACE,
    # not its full volume -- so even a fully-visible target naturally
    # occupies a modest (not near-1.0) fraction of a 3D bounding-box grid.
    # This is realistic depth-camera behavior, not a bug. The threshold
    # below just checks "clearly non-trivial", not "near total occupancy".
    assert pv > 0.1, f"Expected non-trivial perceived visibility for a fully visible target, got {pv}"

    # Case 2: empty mask (no detection) -> integration skipped, not crashed
    empty_mask = np.zeros((480, 640), dtype=np.uint8)
    fusion2 = MaskedTSDFFusion()
    result2 = fusion2.integrate_frame(color, depth, empty_mask, intr)
    assert not result2["integrated"], "Expected integration to be skipped for empty mask"
    print(f"Case 2 (empty mask correctly skipped) OK: {result2['note']}")
    pv2 = fusion2.compute_perceived_visibility(target_center, target_half)
    assert pv2 == 0.0, f"Expected 0.0 perceived visibility with no frames integrated, got {pv2}"
    print(f"Case 2b (perceived_visibility=0 with no integrated frames) OK: {pv2}")

    # Case 3: heavy depth dropout (glass-style sensor failure) -> low but
    # not necessarily zero perceived visibility, must not crash
    color3, depth3, mask3 = make_synthetic_depth_frame(
        640, 480, intr, target_center, target_half, dropout_fraction=0.95, seed=1
    )
    fusion3 = MaskedTSDFFusion()
    result3 = fusion3.integrate_frame(color3, depth3, mask3, intr)
    print(f"Case 3 (heavy depth dropout, glass-style) OK: {result3}")

    # Case 4: multi-frame re-fusion (C5's active re-observation) -- integrating
    # a second, less-occluded view into the SAME volume should not decrease
    # perceived_visibility versus the single-frame case.
    occluder_box = (u0_est := 280, 0, 360, 480)  # blacks out most of the target
    color_occ, depth_occ, mask_occ = make_synthetic_depth_frame(
        640, 480, intr, target_center, target_half, occluder_uv_box=occluder_box
    )
    fusion4 = MaskedTSDFFusion()
    fusion4.integrate_frame(color_occ, depth_occ, mask_occ, intr)
    pv_before = fusion4.compute_perceived_visibility(target_center, target_half, grid_resolution=8)

    # second, unoccluded viewpoint fused into the SAME volume
    fusion4.integrate_frame(color, depth, mask, intr)
    pv_after = fusion4.compute_perceived_visibility(target_center, target_half, grid_resolution=8)

    print(f"Case 4 (multi-frame re-fusion / C5 recovery) OK: "
          f"pv_before_second_view={pv_before:.4f}, pv_after={pv_after:.4f}")
    assert pv_after >= pv_before, (
        f"Expected re-fusion with an unoccluded second view to not decrease "
        f"perceived visibility (before={pv_before}, after={pv_after})"
    )

    print("\nAll tsdf_fusion.py self-tests passed.")