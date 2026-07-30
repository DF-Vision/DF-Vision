"""
debug_aruco.py

Diagnostic: detects ANY ArUco markers in the last captured frame,
regardless of expected ID, and draws them for visual inspection.
Use this when detect_marker_pose() fails to find a specific marker_id --
tells you whether NOTHING was detected, or something WAS detected but
with an unexpected ID.
"""

import cv2
import numpy as np
import sys

sys.path.insert(0, ".")
from orbbec_capture import open_pipeline_with_d2c, capture_aligned_rgbd, make_align_filter

pipeline = open_pipeline_with_d2c()
align_filter = make_align_filter()
try:
    color, depth_mm, intr_dict = capture_aligned_rgbd(pipeline, align_filter)

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, rejected = detector.detectMarkers(gray)

    print(f"Detected IDs: {ids.flatten().tolist() if ids is not None else 'NONE'}")
    print(f"Number of rejected candidates: {len(rejected)}")

    vis = color.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    cv2.imwrite("aruco_debug_detected.png", vis)
    print("Saved aruco_debug_detected.png")

    vis2 = color.copy()
    cv2.aruco.drawDetectedMarkers(vis2, rejected, borderColor=(0, 0, 255))
    cv2.imwrite("aruco_debug_rejected.png", vis2)
    print("Saved aruco_debug_rejected.png (rejected candidates in red)")

finally:
    pipeline.stop()
