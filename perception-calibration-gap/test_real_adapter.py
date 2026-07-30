"""
test_real_hardware_adapter.py

No real camera exists in this sandbox, so this validates the adapter the
only honest way possible without hardware: synthesize a photorealistic-
enough image of an ArUco marker rendered at a KNOWN 3D pose (via OpenCV's
own projectPoints + warpPerspective, not via the adapter itself -- that
would be circular), run the real detect_markers()/rvec_tvec_to_pose()
pipeline on that synthetic image, and check the recovered pose matches the
known ground truth within a tight tolerance.

This is the same spirit as hand_calc_validation.py: an independent
reference (here, OpenCV's own projection math, called directly) checked
against the pipeline under test (here, our adapter's detection + solvePnP
call), not the adapter checking itself.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gazebo"))

import numpy as np
import cv2
import cv2.aruco as aruco

from real_hardware_adapter import (
    CameraCalibration, detect_markers, rvec_tvec_to_pose,
    build_scene_from_detections, ARUCO_DICTIONARY, _marker_object_points,
)
from occlusion_estimator import ObjectGeometry, compute_true_visibility


def synthesize_marker_frame(marker_id: int, marker_size_m: float,
                             rvec_true: np.ndarray, tvec_true: np.ndarray,
                             calib: CameraCalibration,
                             img_w: int = 1280, img_h: int = 960,
                             marker_px: int = 300) -> np.ndarray:
    """Renders a synthetic photo: generate the marker's bitmap, project its
    4 known corners (in the marker's local frame) to image space using the
    GROUND-TRUTH pose via cv2.projectPoints (independent of the adapter
    under test), then warp the bitmap onto those projected corners."""
    marker_img = aruco.generateImageMarker(ARUCO_DICTIONARY, marker_id, marker_px)
    marker_img_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    obj_pts = _marker_object_points(marker_size_m)  # (4,3), matches adapter's own convention
    img_pts, _ = cv2.projectPoints(obj_pts, rvec_true, tvec_true,
                                    calib.camera_matrix, calib.dist_coeffs)
    img_pts = img_pts.reshape(4, 2).astype(np.float32)

    src_pts = np.array([[0, 0], [marker_px, 0], [marker_px, marker_px], [0, marker_px]],
                        dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_pts, img_pts)

    canvas = np.full((img_h, img_w, 3), 40, dtype=np.uint8)  # dark gray background
    warped_marker = cv2.warpPerspective(marker_img_bgr, H, (img_w, img_h),
                                         borderMode=cv2.BORDER_TRANSPARENT)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, img_pts.astype(np.int32), 255)
    canvas[mask > 0] = warped_marker[mask > 0]

    return canvas


def run_pose_recovery_test():
    print("=" * 70)
    print("Test 1: pose recovery accuracy (synthetic frame, known ground truth)")
    print("=" * 70)

    calib = CameraCalibration(fx=800.0, fy=800.0, cx=640.0, cy=480.0,
                               dist_coeffs=np.zeros(5))
    marker_size_m = 0.045  # 4.5cm, e.g. the glass/remote marker size

    test_cases = [
        # (rvec, tvec) -- straight-on view, moderate depth
        (np.array([[0.0], [0.0], [0.0]]), np.array([[0.0], [0.0], [0.6]])),
        # lateral offset + a bit of yaw
        (np.array([[0.0], [0.3], [0.0]]), np.array([[0.08], [0.02], [0.5]])),
        # closer, larger yaw
        (np.array([[0.0], [0.6], [0.0]]), np.array([[-0.05], [0.0], [0.35]])),
    ]

    all_ok = True
    for i, (rvec_true, tvec_true) in enumerate(test_cases):
        frame = synthesize_marker_frame(marker_id=3, marker_size_m=marker_size_m,
                                         rvec_true=rvec_true, tvec_true=tvec_true,
                                         calib=calib)
        detections = detect_markers(frame, calib, marker_sizes_m={3: marker_size_m})

        if 3 not in detections:
            print(f"  Case {i+1}: FAIL -- marker not detected at all")
            all_ok = False
            continue

        pose = rvec_tvec_to_pose(detections[3]["rvec"], detections[3]["tvec"])

        true_x, true_y, true_z = tvec_true[0, 0], tvec_true[1, 0], tvec_true[2, 0]
        pos_err = np.sqrt((pose.x - true_x)**2 + (pose.y - true_y)**2 + (pose.z - true_z)**2)

        R_true, _ = cv2.Rodrigues(rvec_true)
        yaw_true = float(np.arctan2(R_true[0, 2], R_true[2, 2]))
        yaw_err = abs(pose.yaw - yaw_true)

        ok = pos_err < 0.005 and yaw_err < 0.02  # 5mm, ~1.1 degree tolerance
        status = "PASS" if ok else "FAIL"
        print(f"  Case {i+1}: {status} -- true=({true_x:.3f},{true_y:.3f},{true_z:.3f}) "
              f"recovered=({pose.x:.4f},{pose.y:.4f},{pose.z:.4f}) "
              f"pos_err={pos_err*1000:.2f}mm  yaw_true={yaw_true:.4f} "
              f"yaw_recovered={pose.yaw:.4f} yaw_err={yaw_err:.4f}rad")
        all_ok = all_ok and ok

    return all_ok


def run_end_to_end_visibility_test():
    print("\n" + "=" * 70)
    print("Test 2: full pipeline -- synthetic 2-marker scene -> real")
    print("compute_true_visibility(), same depth-ordering logic hand_calc_validation.py checks")
    print("=" * 70)

    calib = CameraCalibration(fx=800.0, fy=800.0, cx=640.0, cy=480.0,
                               dist_coeffs=np.zeros(5))

    # Target: apple-ish object, id=0, marker 3.5cm, at depth 0.5m, dead center
    target_marker_size = 0.035
    target_rvec = np.array([[0.0], [0.0], [0.0]])
    target_tvec = np.array([[0.0], [0.0], [0.5]])

    # Occluder: id=5 (occluder_A), 5cm marker, CLOSER than target (z=0.35 < 0.5)
    # and laterally offset just enough to cause PARTIAL occlusion -- mirrors
    # hand_calc_validation.py's own "apple_partial_occ_01" template scene.
    occluder_marker_size = 0.05
    occluder_rvec = np.array([[0.0], [0.0], [0.0]])
    occluder_tvec = np.array([[0.02], [0.0], [0.35]])

    frame1 = synthesize_marker_frame(0, target_marker_size, target_rvec, target_tvec, calib)
    frame2 = synthesize_marker_frame(5, occluder_marker_size, occluder_rvec, occluder_tvec, calib)
    # composite: paste marker 5's non-background pixels onto frame1 (simulates
    # both markers visible in one real frame)
    combined = frame1.copy()
    diff_mask = np.any(frame2 != 40, axis=-1)
    combined[diff_mask] = frame2[diff_mask]

    marker_sizes = {0: target_marker_size, 5: occluder_marker_size}
    detections = detect_markers(combined, calib, marker_sizes)
    print(f"  Detected marker ids: {sorted(detections.keys())}")

    target_geom = ObjectGeometry(width=0.08, height=0.08, depth=0.08)  # apple-ish AABB
    occluder_geom = ObjectGeometry(width=0.15, height=0.15, depth=0.02)  # occluder panel-ish

    scene = build_scene_from_detections(
        detections, target_marker_id=0, target_geom=target_geom,
        occluder_marker_id=5, occluder_geom=occluder_geom,
    )
    if scene is None:
        print("  FAIL: target not detected, no scene assembled")
        return False

    target_pose, t_geom, occluder_pose, o_geom = scene
    visibility = compute_true_visibility(
        target_pose, t_geom, occluder_pose, o_geom, calib.as_intrinsics_dict()
    )
    print(f"  Adapter-assembled scene -> compute_true_visibility() = {visibility:.4f}")
    ok = 0.0 < visibility < 1.0  # expect partial occlusion, matching the scene's design
    print(f"  Expected partial occlusion (0 < v < 1): {'PASS' if ok else 'FAIL'}")

    # No-occluder case: only the target marker present
    detections_no_occ = {k: v for k, v in detections.items() if k == 0}
    scene2 = build_scene_from_detections(
        detections_no_occ, target_marker_id=0, target_geom=target_geom,
        occluder_marker_id=5, occluder_geom=occluder_geom,
    )
    _, _, occ_pose2, occ_geom2 = scene2
    visibility2 = compute_true_visibility(
        scene2[0], scene2[1], occ_pose2, occ_geom2, calib.as_intrinsics_dict()
    )
    print(f"  No-occluder-detected fallback -> compute_true_visibility() = {visibility2:.4f} "
          f"(expect 1.0)")
    ok2 = abs(visibility2 - 1.0) < 1e-6

    return ok and ok2


if __name__ == "__main__":
    r1 = run_pose_recovery_test()
    r2 = run_end_to_end_visibility_test()
    print("\n" + "=" * 70)
    if r1 and r2:
        print("ALL TESTS PASSED -- adapter's detection+pose math is verified against an")
        print("independent ground truth, and its output is consumable by the REAL,")
        print("unmodified compute_true_visibility() with sensible results.")
    else:
        print("SOME TESTS FAILED -- do not wire this into run_trial.py yet.")
    print("=" * 70)