"""
replay_source.py

On-disk layout for `--replay` mode (capture_and_gap.py), and the reader/
writer pair that share it. The SAME layout a real capture session could
record (write_replay_trial is not fixture-only -- it's the general
writer; generate_replay_fixture.py just calls it with SYNTHETIC data).

Layout
------
<replay_dir>/
    manifest.json                  # {"trials": [{"trial_id": int, "subdir": str,
                                    #              "occluder_present": bool}, ...]}
    <subdir>/
        color.png                  # BGR, any cv2-readable format
        depth_mm.png                # uint16 PNG, millimeters, 0=invalid (see depth_io.py)
        intrinsics.json              # {"fx": float, "fy": float, "cx": float, "cy": float}
        meta.json                    # {"occluder_present": bool, "trial_id": int,
                                      #  "inject_fault": null | "missing_depth" |
                                      #  "corrupt_depth" | "implausible_depth" | "no_marker"}
        (depth_mm.png may be ABSENT for a deliberately faulted "missing_depth" trial)

A trial subdir is intentionally allowed to be faulty (missing/corrupt
depth, no detectable marker) -- ReplayFrameSource does not pre-filter
these out; capture_and_gap.py's per-trial try/except handles them and
flags the row, per C2 criterion 7. This module raises specific,
catchable exceptions rather than returning sentinel values, so the
caller's except clauses stay meaningful.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from depth_io import read_depth_mm_png


class ReplayFrameError(RuntimeError):
    """Raised by ReplayFrameSource.load_frame() for any trial whose saved
    frame can't be loaded (missing file, corrupt PNG, wrong dtype/shape)."""


def write_replay_trial(replay_dir: str, trial_id: int, color_bgr: np.ndarray,
                        depth_mm_uint16: np.ndarray, intrinsics_dict: dict,
                        occluder_present: bool, inject_fault: Optional[str] = None) -> str:
    """Writes one trial's frames to <replay_dir>/trial_%04d/ in the
    documented layout. inject_fault, if given, deliberately corrupts or
    omits what's written (used only by generate_replay_fixture.py to
    build test fixtures for C2 criterion 7 -- a real recording session
    would never pass inject_fault). Returns the subdir path."""
    subdir_name = f"trial_{trial_id:04d}"
    subdir = os.path.join(replay_dir, subdir_name)
    os.makedirs(subdir, exist_ok=True)

    cv2.imwrite(os.path.join(subdir, "color.png"), color_bgr)

    if inject_fault == "missing_depth":
        pass  # deliberately do not write depth_mm.png
    elif inject_fault == "corrupt_depth":
        with open(os.path.join(subdir, "depth_mm.png"), "wb") as f:
            f.write(b"NOT A REAL PNG FILE \x00\x01\x02")
    elif inject_fault == "implausible_depth":
        huge = np.full_like(depth_mm_uint16, 50000, dtype=np.uint16)  # 50m -- implausible
        cv2.imwrite(os.path.join(subdir, "depth_mm.png"), huge)
    else:
        cv2.imwrite(os.path.join(subdir, "depth_mm.png"), depth_mm_uint16.astype(np.uint16))

    with open(os.path.join(subdir, "intrinsics.json"), "w") as f:
        json.dump(intrinsics_dict, f)

    with open(os.path.join(subdir, "meta.json"), "w") as f:
        json.dump({
            "trial_id": trial_id,
            "occluder_present": occluder_present,
            "inject_fault": inject_fault,
        }, f)

    return subdir


def write_manifest(replay_dir: str, trial_specs: list) -> None:
    """trial_specs: list of {"trial_id": int, "subdir": str, "occluder_present": bool}."""
    with open(os.path.join(replay_dir, "manifest.json"), "w") as f:
        json.dump({"trials": trial_specs}, f, indent=2)


class ReplayFrameSource:
    """Iterates trials from a replay directory's manifest.json, in order.
    Each .load_frame(trial_spec) returns (color_bgr, depth_mm, intrinsics_dict)
    -- the SAME shape/units orbbec_capture.capture_aligned_rgbd() returns
    for live capture (depth_mm is a millimeter-valued array, NOT yet
    meters-converted -- capture_and_gap.py does that conversion in the one
    place depth_io.mm_array_to_meters is called, same as before this
    refactor)."""

    def __init__(self, replay_dir: str):
        self.replay_dir = Path(replay_dir)
        manifest_path = self.replay_dir / "manifest.json"
        if not manifest_path.exists():
            raise ReplayFrameError(f"No manifest.json found in replay dir: {replay_dir}")
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.trials = manifest["trials"]

    def __len__(self):
        return len(self.trials)

    def __iter__(self):
        return iter(self.trials)

    def load_frame(self, trial_spec: dict):
        subdir = self.replay_dir / trial_spec["subdir"]
        color_path = subdir / "color.png"
        depth_path = subdir / "depth_mm.png"
        intr_path = subdir / "intrinsics.json"

        if not color_path.exists():
            raise ReplayFrameError(f"Missing color.png for trial {trial_spec['trial_id']} ({subdir})")
        color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise ReplayFrameError(f"Corrupt color.png for trial {trial_spec['trial_id']} ({subdir})")

        if not depth_path.exists():
            raise ReplayFrameError(f"Missing depth_mm.png for trial {trial_spec['trial_id']} ({subdir})")
        try:
            depth_mm = read_depth_mm_png(str(depth_path))
        except RuntimeError as e:
            raise ReplayFrameError(f"Trial {trial_spec['trial_id']} ({subdir}): {e}") from e

        if not intr_path.exists():
            raise ReplayFrameError(f"Missing intrinsics.json for trial {trial_spec['trial_id']} ({subdir})")
        with open(intr_path) as f:
            intrinsics_dict = json.load(f)

        return color_bgr, depth_mm, intrinsics_dict
