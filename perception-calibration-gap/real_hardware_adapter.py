"""
real_hardware_adapter.py

The one new piece of software Day 7-9 (Checkpoint 2) actually needs: reads
ArUco marker detections from the oracle camera and converts them into the
exact same Pose / ObjectGeometry objects occlusion_estimator.py already
expects. Nothing in occlusion_estimator.py, tsdf_fusion.py, or
gap_computer.py changes — this is the "zero-dependency swap point" those
modules were designed around.

Coordinate convention (must match occlusion_estimator.py exactly):
  Pose(x, y, z, yaw) is in the CAMERA's own frame, with z = depth along the
  camera's viewing axis (the same convention _project_aabb_to_image() uses:
  u = fx*(x/z)+cx). solvePnP's tvec, when object points are given in the
  marker's own local frame and the camera matrix is the oracle camera's
  real intrinsics, IS already in the camera frame -- no extra transform
  needed for position. yaw is extracted from the rotation matrix as
  rotation about the camera's vertical (Y) axis, since occlusion_estimator's
  Pose only carries yaw (a documented simplification -- full 6DOF is
  discarded on purpose, matching the AABB-projection model's own scope).

marker_to_object_offset (already a field on ObjectGeometry) is NOT applied
here -- this adapter returns the MARKER's pose. The offset is applied
downstream inside occlusion_estimator._project_aabb_to_image(), exactly as
it already is for the glass's marker-on-a-tab case. Applying it twice would
silently double-count the offset -- do not add offset math to this file.

Edge cases handled:
  - Marker not detected in a given frame -> returns None for that marker id,
    caller must decide whether to skip the trial or retry, never silently
    substitute a stale/last-known pose (that would corrupt Gap for exactly
    the trials where detection is hardest, i.e. high occlusion -- the
    trials that matter most).
  - Multiple markers with the same id in frame (shouldn't happen given the
    manifest's unique-id-per-role rule, but checked defensively) -> raises,
    doesn't silently pick one.
  - Degenerate solvePnP (e.g. marker nearly edge-on) -> passes through
    whatever cv2 returns; occlusion_estimator.py already guards its own
    downstream math (z<=0 checks, degenerate-projection checks), so this
    adapter does not need to duplicate that guarding.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2
import cv2.aruco as aruco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gazebo"))
from occlusion_estimator import Pose, ObjectGeometry  # noqa: E402


@dataclass
class CameraCalibration:
    """Real oracle-camera intrinsics + distortion, from your extrinsic
    calibration pass (Day 1, step 5). fx/fy/cx/cy match the dict shape
    occlusion_estimator.compute_true_visibility() already expects."""
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray  # (5,) or (k,) as returned by cv2.calibrateCamera

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ])

    def as_intrinsics_dict(self) -> dict:
        """Matches occlusion_estimator.compute_true_visibility()'s expected
        camera_intrinsics dict shape exactly."""
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}


ARUCO_DICTIONARY = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)


def _marker_object_points(marker_size_m: float) -> np.ndarray:
    """Object points for a square marker in its own local frame, matching
    OpenCV's ArUco corner ordering (clockwise from top-left), Z=0 plane."""
    h = marker_size_m / 2.0
    return np.array([
        [-h,  h, 0.0],
        [ h,  h, 0.0],
        [ h, -h, 0.0],
        [-h, -h, 0.0],
    ], dtype=np.float32)


def detect_markers(frame: np.ndarray, calib: CameraCalibration,
                    marker_sizes_m: dict) -> dict:
    """
    Detects all ArUco markers in one frame and solves each one's pose via
    solvePnP (not the deprecated estimatePoseSingleMarkers convenience
    function, for forward compatibility across OpenCV versions).

    Parameters
    ----------
    frame : (H, W, 3) uint8 BGR image from the oracle camera.
    calib : CameraCalibration for the oracle camera.
    marker_sizes_m : dict mapping marker_id -> physical side length in
        meters (from generate_markers.py's manifest.csv, converted from cm).

    Returns
    -------
    dict[int, dict] mapping marker_id -> {'rvec', 'tvec', 'corners'}, one
    entry per DETECTED marker only. A marker_id present in marker_sizes_m
    but not in the returned dict means it was not detected this frame --
    callers must handle that explicitly (see module docstring).
    """
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(ARUCO_DICTIONARY, detector_params)
    corners_list, ids, _rejected = detector.detectMarkers(frame)

    results = {}
    if ids is None:
        return results

    seen_ids = set()
    for corners, marker_id_arr in zip(corners_list, ids):
        marker_id = int(marker_id_arr[0])
        if marker_id in seen_ids:
            raise ValueError(
                f"Marker id {marker_id} detected more than once in a single "
                f"frame -- manifest.csv guarantees unique ids per role, so "
                f"this indicates a false-positive detection or a duplicated "
                f"physical marker. Investigate before trusting this frame."
            )
        seen_ids.add(marker_id)

        if marker_id not in marker_sizes_m:
            continue  # a marker we don't have a known physical size for -- ignore, don't guess

        obj_pts = _marker_object_points(marker_sizes_m[marker_id])
        img_pts = corners[0].astype(np.float32)  # (4, 2)

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, calib.camera_matrix, calib.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,  # designed for planar square markers
        )
        if not ok:
            continue  # solvePnP failed for this marker this frame -- treat as not-detected

        results[marker_id] = {"rvec": rvec, "tvec": tvec, "corners": img_pts}

    return results


def rvec_tvec_to_pose(rvec: np.ndarray, tvec: np.ndarray) -> Pose:
    """
    Converts a solvePnP result to occlusion_estimator.Pose(x, y, z, yaw).

    Position: tvec is already in the camera's own frame (meters), matching
    Pose's convention exactly -- no transform needed.

    Yaw: extracted from the rotation matrix as rotation about the camera's
    vertical (Y) axis. occlusion_estimator.py's Pose only carries yaw (a
    documented simplification of full 6DOF, matching the AABB-projection
    model's own scope) -- so roll/pitch information from the marker's
    rotation is deliberately discarded here, not because it's unavailable,
    but because nothing downstream uses it.
    """
    R, _ = cv2.Rodrigues(rvec)
    # yaw about the camera's Y axis, standard extraction from a rotation matrix
    yaw = float(np.arctan2(R[0, 2], R[2, 2]))

    x, y, z = float(tvec[0, 0]), float(tvec[1, 0]), float(tvec[2, 0])
    return Pose(x=x, y=y, z=z, yaw=yaw)


def build_scene_from_detections(
    detections: dict,
    target_marker_id: int,
    target_geom: ObjectGeometry,
    occluder_marker_id: Optional[int],
    occluder_geom: Optional[ObjectGeometry],
    far_behind_z: float = 10.0,
):
    """
    Assembles (target_pose, target_geom, occluder_pose, occluder_geom) ready
    to pass straight into occlusion_estimator.compute_true_visibility().

    Returns None (the whole scene, not a partial one) if the TARGET marker
    was not detected -- a scene without the target isn't a visibility
    question, matching the same "undefined scene" logic
    occlusion_estimator.py itself already applies for an unprojectable
    target. If the OCCLUDER marker is present in marker_sizes_m but wasn't
    detected this frame (e.g. currently swung fully out of frame at the
    0%-occlusion condition), this returns a degenerate, far-behind dummy
    occluder rather than None -- matching hand_calc_validation.py's own
    no-occluder-scene convention, since compute_true_visibility() does not
    accept None for the occluder.
    """
    if target_marker_id not in detections:
        return None

    target_pose = rvec_tvec_to_pose(
        detections[target_marker_id]["rvec"], detections[target_marker_id]["tvec"]
    )

    if occluder_marker_id is not None and occluder_marker_id in detections:
        occluder_pose = rvec_tvec_to_pose(
            detections[occluder_marker_id]["rvec"], detections[occluder_marker_id]["tvec"]
        )
        occ_geom = occluder_geom
    else:
        # No occluder detected this frame -- degenerate, far-behind dummy,
        # matching hand_calc_validation.call_real_pipeline()'s convention.
        occluder_pose = Pose(x=0.0, y=0.0, z=target_pose.z + far_behind_z)
        occ_geom = ObjectGeometry(width=0.0, height=0.0, depth=0.0)

    return target_pose, target_geom, occluder_pose, occ_geom