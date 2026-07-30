"""
orbbec_capture.py

Thin adapter around pyorbbecsdk: opens the Gemini 2, enables D2C
(depth-to-color) alignment, and returns one synchronized RGB-D frame in
the exact format tsdf_fusion.py expects (mirrors mujoco_render.py's role
for the sim side -- this is the same adapter pattern, just for real
hardware).

Do NOT modify tsdf_fusion.py, occlusion_estimator.py, or gap_computer.py
to accommodate this file. If something doesn't fit, fix it here.

STATUS: adapt against your actual pyorbbecsdk 2.1.1 API before trusting --
verify method names against `3.advanced.hw_d2c_align` in the official
examples first. This is a structural template.
"""

import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, AlignFilter, OBStreamType


def open_pipeline_with_d2c():
    """Returns a started Pipeline configured for aligned color+depth."""
    config = Config()
    pipeline = Pipeline()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_video_stream_profile(1280, 720, OBFormat.MJPG, 30)
    config.enable_stream(color_profile)

    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_video_stream_profile(1280, 800, OBFormat.Y16, 30)
    config.enable_stream(depth_profile)

    pipeline.start(config)
    return pipeline


def capture_aligned_rgbd(pipeline, align_filter, timeout_ms: int = 5000,
                          max_attempts: int = 10):
    """
    Returns (color_bgr, depth_mm, intrinsics_dict).
    Retries several times since D2C alignment commonly fails on the first
    few frames while the camera's auto-exposure/buffers settle -- this is
    expected startup behavior, not necessarily a real fault.
    """
    last_error = None
    for attempt in range(max_attempts):
        frames = pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            last_error = "wait_for_frames returned None"
            print(f"  [attempt {attempt+1}/{max_attempts}] {last_error}, retrying...")
            continue

        color_frame_raw = frames.get_color_frame()
        depth_frame_raw = frames.get_depth_frame()
        if color_frame_raw is None or depth_frame_raw is None:
            last_error = (f"pre-align frames missing: color={color_frame_raw is not None}, "
                          f"depth={depth_frame_raw is not None}")
            print(f"  [attempt {attempt+1}/{max_attempts}] {last_error}, retrying...")
            continue

        aligned_frames = align_filter.process(frames)
        if aligned_frames is None:
            last_error = "align_filter.process() returned None"
            print(f"  [attempt {attempt+1}/{max_attempts}] {last_error}, retrying...")
            continue

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            last_error = (f"post-align frames missing: color={color_frame is not None}, "
                          f"depth={depth_frame is not None}")
            print(f"  [attempt {attempt+1}/{max_attempts}] {last_error}, retrying...")
            continue

        # Success -- got a real aligned frame pair
        color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        import cv2
        color_bgr = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
        if color_bgr is None:
            last_error = "cv2.imdecode() returned None (corrupt JPEG frame)"
            print(f"  [attempt {attempt+1}/{max_attempts}] {last_error}, retrying...")
            continue

        width = depth_frame.get_width()
        height = depth_frame.get_height()
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(height, width)
        depth_scale = depth_frame.get_depth_scale()
        depth_mm = depth_data.astype(np.float32) * depth_scale

        video_stream_profile = color_frame.get_stream_profile().as_video_stream_profile()
        intr = video_stream_profile.get_intrinsic()
        intrinsics_dict = {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy}

        print(f"  Succeeded on attempt {attempt+1}/{max_attempts}")
        return color_bgr, depth_mm, intrinsics_dict

    raise RuntimeError(f"Failed to get aligned frames after {max_attempts} attempts. "
                       f"Last error: {last_error}")


def make_align_filter():
    return AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)


if __name__ == "__main__":
    # Self-test: capture one frame, print shapes/ranges, save to disk for
    # visual inspection. Run this FIRST, alone, before writing anything
    # downstream -- exactly like MuJoCo Day 1's "render one frame, inspect it."
    pipeline = open_pipeline_with_d2c()
    align_filter = make_align_filter()
    try:
        color, depth_mm, intr = capture_aligned_rgbd(pipeline, align_filter)
        print(f"color shape/dtype: {color.shape} {color.dtype}")
        print(f"depth shape/dtype: {depth_mm.shape} {depth_mm.dtype}")
        print(f"depth range (mm): min={depth_mm[depth_mm>0].min():.1f} "
              f"max={depth_mm.max():.1f} mean={depth_mm[depth_mm>0].mean():.1f}")


        h, w = depth_mm.shape
        print(f"depth at center: {depth_mm[h//2, w//2]:.1f}mm")

        print(f"intrinsics: {intr}")
        assert color.shape[:2] == depth_mm.shape, (
            "color/depth shape mismatch after D2C alignment -- alignment "
            "did not actually run, or resolutions weren't matched"
        )
        import cv2
        cv2.imwrite("orbbec_capture_test_color.png", color)
        cv2.imwrite("orbbec_capture_test_depth_vis.png",
                     cv2.applyColorMap(cv2.convertScaleAbs(depth_mm, alpha=0.05), cv2.COLORMAP_JET))
        print("\nSaved orbbec_capture_test_color.png and orbbec_capture_test_depth_vis.png")
        print("Self-test PASSED -- inspect the PNGs before proceeding to Step 2.")
    finally:
        pipeline.stop()