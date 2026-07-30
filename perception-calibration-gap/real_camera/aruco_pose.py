"""
aruco_pose.py

Detects an ArUco marker in a real captured color frame and returns its
pose in the camera's own frame via solvePnP -- independent of any
detector/YOLO output, matching occlusion_estimator.py's independence
requirement.
"""

import cv2
import numpy as np


def detect_marker_pose(color_bgr, marker_id: int, marker_size_m: float,
                        camera_matrix, dist_coeffs=None):
    """
    Returns (xyz, corners) for the given marker_id, or raises if not found.

    xyz : (3,) array, meters, OpenCV camera frame (x-right, y-down, z-forward)
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or marker_id not in ids.flatten():
        raise RuntimeError(f"marker_id={marker_id} not detected this frame")

    idx = list(ids.flatten()).index(marker_id)
    marker_corners = corners[idx]

    obj_points = np.array([
        [-marker_size_m / 2,  marker_size_m / 2, 0],
        [ marker_size_m / 2,  marker_size_m / 2, 0],
        [ marker_size_m / 2, -marker_size_m / 2, 0],
        [-marker_size_m / 2, -marker_size_m / 2, 0],
    ], dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(obj_points, marker_corners[0], camera_matrix, dist_coeffs)
    if not ok:
        raise RuntimeError(f"solvePnP failed for marker_id={marker_id}")

    xyz = tvec.flatten()
    return xyz, marker_corners


def build_camera_matrix(intrinsics_dict: dict) -> np.ndarray:
    """Builds the 3x3 camera matrix solvePnP needs, from the same
    intrinsics_dict format used elsewhere in this project."""
    fx, fy = intrinsics_dict["fx"], intrinsics_dict["fy"]
    cx, cy = intrinsics_dict["cx"], intrinsics_dict["cy"]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


if __name__ == "__main__":
    # Self-test: requires ONE real captured frame with a real printed
    # marker (id=0, apple, 3.5cm per manifest.csv) visible in it. Run
    # orbbec_capture.py first, hold the apple marker in view, then run this.
    import sys
    sys.path.insert(0, ".")
    from orbbec_capture import open_pipeline_with_d2c, capture_aligned_rgbd, make_align_filter

    pipeline = open_pipeline_with_d2c()
    align_filter = make_align_filter()
    try:
        color, depth_mm, intr_dict = capture_aligned_rgbd(pipeline, align_filter)

        cv2.imwrite("aruco_test_frame.png", color)
        print("Saved aruco_test_frame.png for inspection")

        camera_matrix = build_camera_matrix(intr_dict)

        MARKER_ID = 0  # apple, per manifest.csv
        MARKER_SIZE_M = 0.035  # 3.5cm, per manifest.csv -- MUST match physical print exactly

        xyz, corners = detect_marker_pose(color, MARKER_ID, MARKER_SIZE_M, camera_matrix)
        print(f"Marker {MARKER_ID} detected at xyz={xyz} (meters, camera frame)")
        print(f"z (depth) = {xyz[2]:.3f}m -- compare against a real tape-measure reading NOW")

        assert xyz[2] > 0, "Depth (z) came back non-positive -- marker pose is wrong"
        print("\nSelf-test PASSED (structurally). Confirm xyz[2] against tape measure before Step 3.")
    finally:
        pipeline.stop()

