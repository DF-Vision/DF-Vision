"""
generate_replay_fixture.py -- SYNTHETIC - NOT A RESULT

Builds a deterministic, on-disk fixture set of saved RGB-D frames in the
same layout a real capture session produces (see replay_source.py's
module docstring), for exercising capture_and_gap.py's `--replay` mode
end to end without a camera. Every artifact this script writes is
labeled SYNTHETIC in path/filename; nothing it produces is evidence for
any real-hardware claim, per HARD RULE 2.

8 trials: 4 good (2 without occluder, 2 with -- mirroring
capture_and_gap.py's own live-session structure), 4 deliberately faulted
(one of each kind C2 criterion 7 requires: missing depth file, corrupt
depth file, implausible depth values, no detectable marker), so a single
fixture directory exercises both the happy path and every documented
failure mode in one `--replay` invocation.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from replay_source import write_replay_trial, write_manifest

RANDOM_STATE = 20260921
IMG_W, IMG_H = 640, 480
TARGET_MARKER_ID = 0
OCCLUDER_MARKER_ID = 5
MARKER_PX = 120

INTRINSICS = {"fx": 600.0, "fy": 600.0, "cx": IMG_W / 2.0, "cy": IMG_H / 2.0}


def _make_marker_image(marker_id: int, size_px: int = MARKER_PX) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)


def _paste_marker(canvas_bgr: np.ndarray, marker_gray: np.ndarray, x0: int, y0: int) -> None:
    h, w = marker_gray.shape
    marker_bgr = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
    canvas_bgr[y0:y0 + h, x0:x0 + w] = marker_bgr


def _make_scene(rng: np.random.Generator, with_occluder: bool, no_marker: bool = False):
    """Returns (color_bgr, depth_mm_uint16) for one synthetic scene."""
    color = np.full((IMG_H, IMG_W, 3), 200, dtype=np.uint8)  # light gray background
    depth_mm = np.full((IMG_H, IMG_W), 500, dtype=np.uint16)  # 0.5m background plane

    if not no_marker:
        target_marker = _make_marker_image(TARGET_MARKER_ID)
        tx0, ty0 = IMG_W // 2 - MARKER_PX // 2, IMG_H // 2 - MARKER_PX // 2
        _paste_marker(color, target_marker, tx0, ty0)
        # target sits slightly closer than background, a plausible small object
        depth_mm[ty0 - 40:ty0 + MARKER_PX + 40, tx0 - 40:tx0 + MARKER_PX + 40] = 480

        if with_occluder:
            occ_marker = _make_marker_image(OCCLUDER_MARKER_ID)
            ox0, oy0 = tx0 + 160, ty0 - 20
            ox0 = min(ox0, IMG_W - MARKER_PX - 1)
            _paste_marker(color, occ_marker, ox0, oy0)
            depth_mm[oy0 - 20:oy0 + MARKER_PX + 20, ox0 - 20:ox0 + MARKER_PX + 20] = 470

    # mild deterministic noise so trials aren't bit-identical to each other
    noise = rng.normal(0, 2.0, size=depth_mm.shape)
    depth_mm = np.clip(depth_mm.astype(np.float64) + noise, 1, 65000).astype(np.uint16)

    return color, depth_mm


def generate(fixture_dir: str, seed: int = RANDOM_STATE) -> None:
    rng = np.random.default_rng(seed)
    out = Path(fixture_dir)
    out.mkdir(parents=True, exist_ok=True)

    trial_specs = []
    trial_id = 0

    # 4 good trials: 2 without occluder, 2 with
    for with_occluder in (False, False, True, True):
        color, depth_mm = _make_scene(rng, with_occluder=with_occluder)
        subdir = write_replay_trial(
            str(out), trial_id, color, depth_mm, INTRINSICS,
            occluder_present=with_occluder, inject_fault=None,
        )
        trial_specs.append({"trial_id": trial_id, "subdir": Path(subdir).name,
                             "occluder_present": with_occluder})
        trial_id += 1

    # 4 deliberately faulted trials, one of each documented kind
    for fault in ("missing_depth", "corrupt_depth", "implausible_depth", "no_marker"):
        color, depth_mm = _make_scene(rng, with_occluder=False, no_marker=(fault == "no_marker"))
        subdir = write_replay_trial(
            str(out), trial_id, color, depth_mm, INTRINSICS,
            occluder_present=False, inject_fault=fault,
        )
        trial_specs.append({"trial_id": trial_id, "subdir": Path(subdir).name,
                             "occluder_present": False})
        trial_id += 1

    write_manifest(str(out), trial_specs)
    print(f"SYNTHETIC fixture written: {out} ({len(trial_specs)} trials, "
          f"4 good + 4 deliberately faulted: missing_depth, corrupt_depth, "
          f"implausible_depth, no_marker). seed={seed}. NOT A RESULT.")


if __name__ == "__main__":
    default_dir = str(Path(__file__).parent / "fixtures" / "replay_synthetic_v1")
    target_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    generate(target_dir)
