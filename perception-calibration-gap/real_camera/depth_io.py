"""
depth_io.py

The ONE place in this codebase that converts on-disk depth (16-bit PNG,
millimeters, matching what `orbbec_capture.py`'s `depth_mm` already is
before capture_and_gap.py's `depth_mm.astype(np.float32) / 1000.0` line)
into in-memory metric depth (float32 meters). Both the live path and the
`--replay` path funnel through `mm_png_to_meters` / `mm_array_to_meters`
so there is exactly one conversion to audit, per C2's acceptance
criterion 3.

On-disk depth format, documented and enforced:
    uint16 PNG, pixel value = depth in MILLIMETERS, 0 = invalid/no-return.
    This matches the Orbbec Gemini 2 SDK's native `depth_scale`-applied
    output (see orbbec_capture.py:88-90) at 1x scale, and is the same
    convention this project's synthetic fixtures (generate_replay_fixture.py)
    write.
"""
from __future__ import annotations

import numpy as np
import cv2

MAX_PLAUSIBLE_MEDIAN_DEPTH_M = 20.0  # task's own stated example threshold --
                                      # no benchtop scene in this project has a
                                      # median depth anywhere near 20m; a value
                                      # this large means someone handed meters-as-
                                      # millimeters (or vice versa) into this path.


class ImplausibleDepthError(ValueError):
    """Raised by validate_depth_meters() when a depth array's median value
    cannot plausibly be meters for this project's benchtop scenes."""


def mm_array_to_meters(depth_mm: np.ndarray) -> np.ndarray:
    """The single mm->m conversion. depth_mm: uint16 or float array, millimeters.
    Returns float32 meters. 0 (invalid-return sentinel) maps to 0.0, not NaN --
    matches capture_and_gap.py's pre-existing treatment of zero-depth pixels."""
    return depth_mm.astype(np.float32) / 1000.0


def read_depth_mm_png(path: str) -> np.ndarray:
    """Reads a 16-bit PNG (millimeters) from disk and returns the raw
    millimeter array (uint16), NOT converted to meters -- used by
    replay_source.py's frame provider, which hands depth_mm downstream to
    tsdf_fusion.py exactly like orbbec_capture.py's live depth_mm (also
    millimeters, also not yet meters-converted at that point in the
    pipeline). Raises RuntimeError on a missing or corrupt file -- callers
    are responsible for catching this and flagging the trial (C2
    criterion 7); this function does not swallow errors itself."""
    depth_mm = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise RuntimeError(f"Failed to read depth PNG (missing or corrupt): {path}")
    if depth_mm.dtype != np.uint16:
        raise RuntimeError(
            f"Depth PNG {path} is dtype={depth_mm.dtype}, expected uint16 "
            f"(millimeters). Refusing to guess a scale."
        )
    return depth_mm


def mm_png_to_meters(path: str) -> np.ndarray:
    """Reads a 16-bit PNG (millimeters) from disk and returns float32
    meters directly -- composition of read_depth_mm_png + mm_array_to_meters,
    kept for callers (and tests) that want the full on-disk-mm -> in-memory-
    meters path in one call."""
    return mm_array_to_meters(read_depth_mm_png(path))


def validate_depth_meters(depth_m: np.ndarray, context: str = "") -> None:
    """Guard: raises ImplausibleDepthError if depth_m's median (over finite,
    positive values only -- zero/invalid pixels excluded) exceeds
    MAX_PLAUSIBLE_MEDIAN_DEPTH_M, or if there are zero valid pixels at all.
    Does not modify or clip depth_m -- validation only, per C2 criterion 3
    ("a guard function... a separate, directly-tested unit, not folded
    silently into the conversion function")."""
    valid = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    if valid.size == 0:
        raise ImplausibleDepthError(
            f"depth array{' (' + context + ')' if context else ''} has zero valid "
            f"(finite, positive) pixels -- cannot validate or use this frame."
        )
    median = float(np.median(valid))
    if median > MAX_PLAUSIBLE_MEDIAN_DEPTH_M:
        raise ImplausibleDepthError(
            f"depth array{' (' + context + ')' if context else ''} has median="
            f"{median:.3f}m, exceeding the {MAX_PLAUSIBLE_MEDIAN_DEPTH_M}m plausibility "
            f"ceiling for this project's benchtop scenes -- likely a millimeters/meters "
            f"unit mismatch upstream."
        )
