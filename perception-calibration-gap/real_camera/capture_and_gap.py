"""
capture_and_gap.py

Real-Hardware Smoke Test -- ties together real Gemini 2 capture +
real ArUco marker poses + the three UNMODIFIED real pipeline modules
(occlusion_estimator.py, tsdf_fusion.py, gap_computer.py) + mock_vla.py's
geometrically-grounded decision + logger.py.

SCOPE: this produces SMOKE-TEST trials only. Never cited as C1/C4
evidence, never merged into the real frozen dataset -- same discipline as
trials_mujoco_smoke.jsonl. Output file name says so explicitly.

Placeholder standing in for a real detector: the target marker's own
detected bounding box, dilated -- NOT a real YOLO/open-vocab detector.
Flag this explicitly in any notes/write-up referencing these trials,
exactly like MuJoCo Day 2's synthetic bbox stand-in.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from integration import prg_client


from integration.camera_intrinsics import CameraIntrinsics as PrgCameraIntrinsics
sys.path.insert(0, "/workspace")
sys.path.insert(0, str(Path(__file__).parent))

from occlusion_estimator import ObjectGeometry, Pose, compute_true_visibility
from tsdf_fusion import MaskedTSDFFusion, CameraIntrinsics
from gap_computer import compute_gap_features
from logger import TrialRecord, TrialLogger
import mock_vla

from orbbec_capture import open_pipeline_with_d2c, capture_aligned_rgbd, make_align_filter
from aruco_pose import detect_marker_pose, build_camera_matrix


# --- CONFIRM THESE AGAINST manifest.csv AND YOUR REAL RULER MEASUREMENTS ---
OBJECT_NAME = "apple"
TARGET_MARKER_ID = 0
TARGET_MARKER_SIZE_M = 0.035
OCCLUDER_MARKER_ID = 5
OCCLUDER_MARKER_SIZE_M = 0.05

# --- REPLACE with real ruler measurements of the actual apple + occluder ---
TARGET_WIDTH_M = 0.075   # PLACEHOLDER, matches APPLE_DIAMETER_CM=7.5 from checkpoint1 script
TARGET_HEIGHT_M = 0.075
TARGET_DEPTH_M = 0.075
TARGET_MARKER_OFFSET_Z_M = -TARGET_DEPTH_M / 2  # marker on front face -> shift to centroid

OCCLUDER_WIDTH_M = 0.10
OCCLUDER_HEIGHT_M = 0.15
OCCLUDER_DEPTH_M = 0.01

LOG_PATH = Path(__file__).parent.parent / "mock-pipeline" / "data" / "phone_screen_code_path_test.jsonl"


def build_bbox_mask_from_corners(image_shape, marker_corners, dilate_px: int = 60):
    """Placeholder detection mask: dilated box around the target marker's
    detected corners. NOT a real detector -- documented stand-in, same
    role as MuJoCo Day 2's synthetic AABB bbox."""
    h, w = image_shape[:2]
    pts = marker_corners[0].reshape(-1, 2)
    x_min, y_min = pts.min(axis=0) - dilate_px
    x_max, y_max = pts.max(axis=0) + dilate_px
    x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
    x_max, y_max = min(w, int(x_max)), min(h, int(y_max))

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 1
    return mask


def run_one_trial(trial_id: int, occluder_present: bool, pipeline, align_filter,
                   logger: TrialLogger, session_baseline_failure_rate: float):
    color, depth_mm, intr_dict = capture_aligned_rgbd(pipeline, align_filter)
    camera_matrix = build_camera_matrix(intr_dict)

    target_xyz, target_corners = detect_marker_pose(
        color, TARGET_MARKER_ID, TARGET_MARKER_SIZE_M, camera_matrix
    )

    # --- true_visibility: real occlusion_estimator.py, unmodified ---
    target_geom = ObjectGeometry(
        width=TARGET_WIDTH_M, height=TARGET_HEIGHT_M, depth=TARGET_DEPTH_M,
        marker_to_object_offset=(0.0, 0.0, TARGET_MARKER_OFFSET_Z_M),
    )
    target_pose = Pose(x=target_xyz[0], y=target_xyz[1], z=target_xyz[2])

    occluder_mask = None

    if occluder_present:
        occ_xyz, occ_corners = detect_marker_pose(
            color,
            OCCLUDER_MARKER_ID,
            OCCLUDER_MARKER_SIZE_M,
            camera_matrix,
        )

        occ_geom = ObjectGeometry(
            width=OCCLUDER_WIDTH_M,
            height=OCCLUDER_HEIGHT_M,
            depth=OCCLUDER_DEPTH_M,
        )

        occ_pose = Pose(
            x=occ_xyz[0],
            y=occ_xyz[1],
            z=occ_xyz[2],
        )

        occluder_mask = build_bbox_mask_from_corners(
            color.shape,
            occ_corners,
        )

    else:
        occ_geom = ObjectGeometry(
            width=0.0,
            height=0.0,
            depth=0.0,
        )

        occ_pose = Pose(
            x=0.0,
            y=0.0,
            z=target_xyz[2] + 10.0,
        )

    true_visibility = compute_true_visibility(target_pose, target_geom, occ_pose, occ_geom, intr_dict)

    # --- perceived_visibility: real tsdf_fusion.py, unmodified ---
    mask = build_bbox_mask_from_corners(color.shape, target_corners)
    intr = CameraIntrinsics(width=color.shape[1], height=color.shape[0],
                             fx=intr_dict["fx"], fy=intr_dict["fy"],
                             cx=intr_dict["cx"], cy=intr_dict["cy"])
    fusion = MaskedTSDFFusion(voxel_length=0.004, sdf_trunc=0.02)
    integ_result = fusion.integrate_frame(color, depth_mm, mask, intr)
    print(f"  [tsdf] {integ_result}")

    # --- DIAGNOSTIC: inspect what actually happened this trial ---
    mask_pixel_count = int(mask.sum())
    depth_in_mask = depth_mm[mask.astype(bool)]
    valid_depth_in_mask = depth_in_mask[depth_in_mask > 0]
    print(f"  [diag] mask_pixels={mask_pixel_count}  "
          f"valid_depth_pixels_in_mask={len(valid_depth_in_mask)}  "
          f"depth_in_mask_range=({valid_depth_in_mask.min() if len(valid_depth_in_mask) else 'N/A'}, "
          f"{valid_depth_in_mask.max() if len(valid_depth_in_mask) else 'N/A'})")
    cv2.imwrite(f"/workspace/debug_trial{trial_id}_color.png", color)
    cv2.imwrite(f"/workspace/debug_trial{trial_id}_mask.png", mask * 255)

    target_center = np.array(target_xyz) + np.array([0, 0, TARGET_MARKER_OFFSET_Z_M])
    perceived_visibility = fusion.compute_perceived_visibility(
        target_center=target_center,
        target_half_extents=np.array([TARGET_WIDTH_M / 2, TARGET_HEIGHT_M / 2, TARGET_DEPTH_M / 2]),
        grid_resolution=8,
    )

    # --- Gap: real gap_computer.py, unmodified, single-step ---
    gap_result = compute_gap_features([perceived_visibility], [true_visibility])

    # --- prg_edge_error: real chain via integration/prg_client.py ---

    depth_m = depth_mm.astype(np.float32) / 1000.0

    all_object_masks = {
        OBJECT_NAME: mask,
    }

    if occluder_mask is not None:
        all_object_masks["occluder"] = occluder_mask

    camera_intrinsics = PrgCameraIntrinsics(
        name="orbbec_gemini2",
        xyz=np.zeros(3),
        rpy=np.zeros(3),
        width=color.shape[1],
        height=color.shape[0],
        fx=intr_dict["fx"],
        fy=intr_dict["fy"],
        cx=intr_dict["cx"],
        cy=intr_dict["cy"],
        clip_near=0.1,
        clip_far=5.0,
    )

    prg_edges_result = prg_client.get_prg_edge_error(
        rgb=color,
        depth=depth_m,
        target_name=OBJECT_NAME,
        all_object_masks=all_object_masks,
        trial_id=str(trial_id),
        camera_intrinsics=camera_intrinsics,
    )

    prg_is_stub = (
        prg_edges_result[0].get("is_stub", True)
        if prg_edges_result
        else True
    )

    prg_error_note = (
        prg_edges_result[0].get("error_note")
        if prg_edges_result
        else None
    )

    # --- decision: real mock_vla.py, geometrically grounded ---
    attempt = mock_vla.execute(
        true_position=target_xyz,
        perceived_visibility=perceived_visibility,
        goal_pose_tolerance=0.02,
        seed=trial_id,
    )

    print(f"  true_vis={true_visibility:.4f}  perceived_vis={perceived_visibility:.4f}  "
          f"Gap={gap_result.gap_initial:.4f}  outcome={attempt.outcome}")

    record = TrialRecord(
        trial_id=trial_id, session_id=0, object_id=OBJECT_NAME,
        session_baseline_failure_rate=session_baseline_failure_rate,
        measured_true_visibility=true_visibility,
        true_occlusion=1.0 - true_visibility,
        clutter_count=0,
        perceived_visibility_initial=perceived_visibility,
        Gap_initial=gap_result.gap_initial, Gap_max=gap_result.gap_max,
        Gap_slope=gap_result.gap_slope,
        failure=0 if attempt.outcome == "success" else 1,
        camera_pose_variant="real_hw_smoke_gemini2",
        outcome=attempt.outcome, failure_type=attempt.failure_type,
        tsdf_fusion_attempted=False, gate_warmup_phase=True, gate_fired=None,
        prg_edges=prg_edges_result,
        prg_is_stub=prg_is_stub,
        prg_error_note=prg_error_note,
    )
    logger.append_trial(record)
    return record


def main():
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    freeze_marker = Path(str(LOG_PATH) + ".FROZEN")
    if freeze_marker.exists():
        freeze_marker.unlink()
    logger = TrialLogger(str(LOG_PATH))

    pipeline = open_pipeline_with_d2c()
    align_filter = make_align_filter()

    try:
        print("=== Real-Hardware Smoke Test: 8 trials ===")
        print("Trial 0-1: zero occlusion (session baseline)")
        baseline_failures = 0
        for i in range(2):
            input(f"\n[trial {i}] Position apple with NO occluder in view, press Enter...")
            r = run_one_trial(i, occluder_present=False, pipeline=pipeline,
                               align_filter=align_filter, logger=logger,
                               session_baseline_failure_rate=0.0)
            baseline_failures += (r.failure == 1)
        session_baseline = baseline_failures / 2

        print(f"\nSession baseline failure rate (trials 0-1): {session_baseline:.3f}")
        print("Trials 2-7: sweep occluder from clear to heavy overlap")

        for i in range(2, 8):
            occluder_present = (i >= 3)  # trial 2: still no occluder as extra sanity check
            prompt = "WITH occluder" if occluder_present else "still WITHOUT occluder"
            input(f"\n[trial {i}] Position apple {prompt} (vary overlap each trial), press Enter...")
            run_one_trial(i, occluder_present=occluder_present, pipeline=pipeline,
                          align_filter=align_filter, logger=logger,
                          session_baseline_failure_rate=session_baseline)

        print(f"\n=== Done. 8 trials written to {LOG_PATH} ===")
        print("REMINDER: these are SMOKE-TEST trials only.")
        print("Detection mask = marker-corner bbox placeholder, NOT a real detector.")
        print("Never cite these as C1/C4 evidence or merge into the real dataset.")

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()