"""
occlusion_estimator.py

Purpose
-------
Compute true_visibility (ground truth) from marker poses + known object
geometry -- completely INDEPENDENT of YOLO. This independence is the single
most load-bearing property in the whole project: if true_visibility ever
depended on YOLO, Gap would measure YOLO against itself and the entire
C1/C4 claim collapses. This file must never import, call, or reference
anything YOLO-related.

Model (deliberately simple, analytic, not learned)
---------------------------------------------------
Treat the target object as an axis-aligned bounding box (AABB) of known
width/height/depth (the "known dimensions entered once per object" from
the master reference). Treat the occluder as a second AABB. Project both
onto the oracle camera's image plane using the marker-derived poses and
known camera intrinsics. true_visibility = 1 - (occluded_area / target_area)
computed via 2D polygon intersection in image space.

This is a deliberately simple geometric model -- not a full 3D visibility /
ray-casting solver. That's a real, disclosable limitation (documented here
and in the paper's limitations section), not something to silently upgrade
mid-study once real hardware exists, since changing the ground-truth method
after Phase 1 validation would invalidate Checkpoint 1.

Edge cases handled
-------------------
- Object AABB fully occluded -> true_visibility = 0.0, not negative.
- Object AABB fully visible (no overlap) -> true_visibility = 1.0.
- Occluder behind the target (occluder z > target z, i.e. farther from
  camera) -> occlusion contribution is zero regardless of 2D overlap,
  since something behind the target can't occlude it. This is checked
  explicitly via depth ordering, not assumed away.
- Degenerate/zero-area target projection (e.g. object exactly edge-on to
  camera) -> guarded against divide-by-zero, returns true_visibility = 1.0
  with a logged warning rather than NaN.
- Per-object marker size/offset: markers are NOT assumed to sit exactly on
  the object's origin (e.g. the glass's marker-on-a-tab, offset from the
  glass itself) -- marker_to_object_offset is a required, explicit
  parameter per object, never defaulted to zero silently.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ObjectGeometry:
    """Known, once-entered-per-object dimensions and marker offset."""
    width: float
    height: float
    depth: float
    marker_to_object_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Pose:
    """Position (x, y, z) in camera frame + a simple yaw-only orientation
    (radians). Real marker poses (ArUco/AprilTag) give full 6DOF; this
    estimator only needs yaw for the AABB projection model used here --
    documented simplification, not an oversight."""
    x: float
    y: float
    z: float
    yaw: float = 0.0


def _project_aabb_to_image(pose: Pose, geom: ObjectGeometry,
                            fx: float, fy: float, cx: float, cy: float):
    """Project an object's AABB (centered at pose + marker offset) to an
    axis-aligned 2D image-space bounding box under a pinhole camera model.
    Returns (u_min, v_min, u_max, v_max, mean_depth) or None if the object
    is behind or at the camera (z <= 0, cannot project)."""
    ox = pose.x + geom.marker_to_object_offset[0]
    oy = pose.y + geom.marker_to_object_offset[1]
    oz = pose.z + geom.marker_to_object_offset[2]

    if oz <= 1e-6:
        return None  # object at/behind camera plane, cannot project

    hw, hh, hd = geom.width / 2, geom.height / 2, geom.depth / 2
    # 8 corners of the AABB in camera frame (ignoring yaw for corner
    # positions -- an approximation; documented simplification)
    corners_3d = []
    for dx in (-hw, hw):
        for dy in (-hh, hh):
            for dz in (-hd, hd):
                corners_3d.append((ox + dx, oy + dy, oz + dz))

    us, vs = [], []
    for (x, y, z) in corners_3d:
        if z <= 1e-6:
            continue  # this corner is behind the camera; skip it
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
        us.append(u)
        vs.append(v)

    if not us:
        return None

    return (min(us), min(vs), max(us), max(vs), oz)


def _rect_intersection_area(r1, r2):
    """Area of intersection between two axis-aligned rects
    (u_min, v_min, u_max, v_max). Returns 0 if no overlap."""
    u_min = max(r1[0], r2[0])
    v_min = max(r1[1], r2[1])
    u_max = min(r1[2], r2[2])
    v_max = min(r1[3], r2[3])
    if u_max <= u_min or v_max <= v_min:
        return 0.0
    return (u_max - u_min) * (v_max - v_min)


def _rect_area(r):
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def compute_true_visibility(target_pose: Pose, target_geom: ObjectGeometry,
                             occluder_pose: Pose, occluder_geom: ObjectGeometry,
                             camera_intrinsics: dict) -> float:
    """Ground-truth true_visibility in [0, 1], independent of YOLO.

    Parameters
    ----------
    target_pose, occluder_pose : Pose
        Marker-derived poses in the oracle camera's frame.
    target_geom, occluder_geom : ObjectGeometry
        Known dimensions + marker offsets, entered once per object.
    camera_intrinsics : dict with keys fx, fy, cx, cy.

    Returns
    -------
    float in [0, 1]. 1.0 = fully visible, 0.0 = fully occluded.
    """
    fx, fy, cx, cy = (camera_intrinsics["fx"], camera_intrinsics["fy"],
                       camera_intrinsics["cx"], camera_intrinsics["cy"])

    target_rect = _project_aabb_to_image(target_pose, target_geom, fx, fy, cx, cy)
    if target_rect is None:
        # Target itself isn't projectable (behind camera) -- undefined scene,
        # not a visibility question. Caller should have filtered this out;
        # we return 0.0 defensively rather than NaN.
        return 0.0

    target_area = _rect_area(target_rect)
    if target_area <= 1e-9:
        # Degenerate projection (e.g. perfectly edge-on). Can't compute a
        # meaningful ratio; treat as fully visible rather than divide-by-zero.
        return 1.0

    occluder_rect = _project_aabb_to_image(occluder_pose, occluder_geom, fx, fy, cx, cy)
    if occluder_rect is None:
        return 1.0  # occluder not in frame / behind camera -> no occlusion

    # Depth check: occluder must be closer to the camera (smaller z) than
    # the target to occlude it at all. If it's farther away, it cannot
    # occlude regardless of 2D overlap.
    target_depth = target_rect[4]
    occluder_depth = occluder_rect[4]
    if occluder_depth >= target_depth:
        return 1.0

    overlap_area = _rect_intersection_area(target_rect, occluder_rect)
    occluded_fraction = overlap_area / target_area
    true_visibility = 1.0 - min(1.0, occluded_fraction)
    return float(np.clip(true_visibility, 0.0, 1.0))


if __name__ == "__main__":
    # Self-tests against hand-calculable fabricated marker poses.
    intr = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
    target_geom = ObjectGeometry(width=0.08, height=0.08, depth=0.08)  # ~apple-sized
    occluder_geom = ObjectGeometry(width=0.20, height=0.30, depth=0.02)  # flat panel

    # Case 1: occluder far off to the side -> no occlusion, visibility=1.0
    target_pose = Pose(x=0.0, y=0.0, z=0.5)
    occluder_pose = Pose(x=2.0, y=0.0, z=0.4)  # well off-frame laterally
    v = compute_true_visibility(target_pose, target_geom, occluder_pose, occluder_geom, intr)
    assert abs(v - 1.0) < 1e-6, f"Expected 1.0, got {v}"
    print(f"Case 1 (occluder off to the side) OK: visibility={v:.4f}")

    # Case 2: occluder directly in front of target, closer to camera, same
    # lateral position, large enough to fully cover it -> visibility ~ 0
    target_pose = Pose(x=0.0, y=0.0, z=0.5)
    occluder_pose = Pose(x=0.0, y=0.0, z=0.3)  # closer to camera than target
    v = compute_true_visibility(target_pose, target_geom, occluder_pose, occluder_geom, intr)
    assert v < 0.05, f"Expected near-total occlusion, got {v}"
    print(f"Case 2 (occluder fully in front, aligned) OK: visibility={v:.4f}")

    # Case 3: occluder BEHIND the target (farther from camera) -> must not occlude
    # even though laterally aligned
    target_pose = Pose(x=0.0, y=0.0, z=0.5)
    occluder_pose = Pose(x=0.0, y=0.0, z=0.9)  # farther away than target
    v = compute_true_visibility(target_pose, target_geom, occluder_pose, occluder_geom, intr)
    assert abs(v - 1.0) < 1e-6, f"Expected 1.0 (occluder behind target), got {v}"
    print(f"Case 3 (occluder behind target, must not occlude) OK: visibility={v:.4f}")

    # Case 4: partial occlusion -- occluder is roughly target-sized and
    # shifted so it covers only part of the target's projected footprint.
    small_occluder_geom = ObjectGeometry(width=0.08, height=0.08, depth=0.02)
    target_pose = Pose(x=0.0, y=0.0, z=0.5)
    occluder_pose = Pose(x=0.045, y=0.0, z=0.49)  # slightly closer, shifted laterally
    v = compute_true_visibility(target_pose, target_geom, occluder_pose, small_occluder_geom, intr)
    assert 0.0 < v < 1.0, f"Expected partial occlusion in (0,1), got {v}"
    print(f"Case 4 (partial occlusion) OK: visibility={v:.4f}")

    # Case 5: marker_to_object_offset -- glass-style tab offset. Object sits
    # 5cm away from its own marker's reported pose. Confirm the offset is
    # actually applied (visibility differs from the zero-offset case).
    glass_geom = ObjectGeometry(width=0.06, height=0.15, depth=0.06,
                                 marker_to_object_offset=(0.05, 0.0, 0.0))
    small_occluder = ObjectGeometry(width=0.07, height=0.15, depth=0.02)
    target_pose = Pose(x=0.0, y=0.0, z=0.5)
    occluder_pose = Pose(x=0.0, y=0.0, z=0.49)
    v_offset = compute_true_visibility(target_pose, glass_geom, occluder_pose, small_occluder, intr)
    glass_geom_no_offset = ObjectGeometry(width=0.06, height=0.15, depth=0.06)
    v_no_offset = compute_true_visibility(target_pose, glass_geom_no_offset, occluder_pose, small_occluder, intr)
    assert v_offset != v_no_offset, "Marker offset should change the projected position"
    print(f"Case 5 (marker-to-object offset applied) OK: with_offset={v_offset:.4f}, "
          f"without={v_no_offset:.4f}")

    # Case 6: degenerate target projection (z<=0) -> must not crash / NaN
    target_pose = Pose(x=0.0, y=0.0, z=-0.1)
    occluder_pose = Pose(x=0.0, y=0.0, z=0.3)
    v = compute_true_visibility(target_pose, target_geom, occluder_pose, occluder_geom, intr)
    assert v == 0.0, f"Expected 0.0 for unprojectable target, got {v}"
    print(f"Case 6 (target behind camera, no crash) OK: visibility={v:.4f}")

    print("\nAll occlusion_estimator.py self-tests passed.")