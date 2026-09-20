"""
tests/test_hand_calc_validation.py

T3 tests: (1) AST-based import-independence check proving
hand_calc_validation.py imports nothing from detector/YOLO code; (2)
synthetic-geometry tests with analytically known visibility; (3) the
new template-refusal guard (refuse_if_still_template).
"""
import ast
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap"))

import numpy as np
import pytest

from hand_calc_validation import (
    AABBScene, hand_calc_visibility, project_aabb_to_rect, rect_overlap_area,
    refuse_if_still_template, write_template_if_missing,
)

HAND_CALC_FILE = PROJECT_ROOT / "perception-calibration-gap" / "hand_calc_validation.py"
FORBIDDEN_MARKERS = ["yolo", "detector", "ultralytics"]


def _make_scene(target_c, target_half, occ_c=None, occ_half=None,
                 fx=600.0, fy=600.0, cx_px=320.0, cy_px=240.0):
    return AABBScene(
        scene_id="synthetic", object_id="apple",
        target_half=np.array(target_half), target_c=np.array(target_c),
        occ_half=np.array(occ_half if occ_half is not None else [0, 0, 0]),
        occ_c=np.array(occ_c if occ_c is not None else [0, 0, 0]),
        fx=fx, fy=fy, cx_px=cx_px, cy_px=cy_px, notes="synthetic test scene",
    )


# --- (1) Import independence -------------------------------------------------

def test_hand_calc_validation_imports_nothing_from_detector_or_yolo():
    tree = ast.parse(HAND_CALC_FILE.read_text(), filename=str(HAND_CALC_FILE))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    lowered = [n.lower() for n in names]
    for marker in FORBIDDEN_MARKERS:
        assert not any(marker in n for n in lowered), (
            f"hand_calc_validation.py imports something matching forbidden "
            f"marker {marker!r}: {names}"
        )


def test_call_real_pipeline_only_imports_occlusion_estimator_not_detector():
    """The one place this file DOES import the real pipeline
    (call_real_pipeline) must only ever reach occlusion_estimator.py --
    checked by reading occlusion_estimator.py's OWN imports too, since a
    transitive detector import there would be just as much a violation."""
    occ_est_file = PROJECT_ROOT / "perception-calibration-gap" / "gazebo" / "occlusion_estimator.py"
    assert occ_est_file.exists()
    tree = ast.parse(occ_est_file.read_text(), filename=str(occ_est_file))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    lowered = [n.lower() for n in names]
    for marker in FORBIDDEN_MARKERS:
        assert not any(marker in n for n in lowered), (
            f"occlusion_estimator.py (imported by hand_calc_validation.py's "
            f"call_real_pipeline) imports something matching forbidden marker "
            f"{marker!r}: {names}"
        )


# --- (2) Synthetic geometry, analytically known visibility -------------------

def test_no_occluder_visibility_is_exactly_one():
    scene = _make_scene(target_c=[0.0, 0.0, 0.5], target_half=[0.035, 0.035, 0.035])
    assert hand_calc_visibility(scene) == pytest.approx(1.0)


def test_occluder_behind_target_visibility_is_one_regardless_of_overlap():
    scene = _make_scene(
        target_c=[0.0, 0.0, 0.5], target_half=[0.035, 0.035, 0.035],
        occ_c=[0.0, 0.0, 0.8], occ_half=[0.05, 0.05, 0.05],  # huge overlap, but BEHIND
    )
    assert hand_calc_visibility(scene) == pytest.approx(1.0)


def test_occluder_fully_covering_target_visibility_is_zero():
    # Occluder much larger than target, same lateral position, in front.
    scene = _make_scene(
        target_c=[0.0, 0.0, 0.5], target_half=[0.02, 0.02, 0.02],
        occ_c=[0.0, 0.0, 0.3], occ_half=[0.5, 0.5, 0.02],
    )
    assert hand_calc_visibility(scene) == pytest.approx(0.0, abs=1e-6)


def test_occluder_half_covering_target_visibility_matches_hand_calc():
    """Analytically constructed so the projected rectangles overlap by
    EXACTLY half the target's area, worked out on paper first:

    Target: half=[0.1,0.1,0.1] at z=1.0, fx=fy=1000, cx=cy=0 -> projected
    rect spans u in [-100,100] (width 200), same for v -- area 40000.
    Occluder: half=[0.1,0.1,0.1] at z=0.5 (in front), offset in x by
    exactly the target's rect width (so its own rect is also 200 wide)
    such that it covers exactly the right half of the target's rect.
    Occluder is closer (z=0.5), so occluder rect width = 1000*0.2/0.5 = 400
    (bigger, as expected for a closer object of the same physical size) --
    to keep the overlap calc simple, use occluder_half small enough that
    its own projected width equals the target's (200), by using half=0.05
    at z=0.5: width = 1000*0.1/0.5 = 200. Centered at x=0.1 (in world,
    z=0.5) -> projected center u = 1000*0.1/0.5 = 200, rect = [100, 300].
    Target rect = [-100, 100]. Overlap = [100,100] -> zero width, not
    useful. Instead center occluder at x=0.05 (world) -> u=100, rect=[0,200].
    Overlap with target [-100,100] is [0,100], width 100 = half of 200.
    Expected visible_fraction = 1 - (100*200)/(200*200) = 1 - 0.5 = 0.5.
    """
    scene = _make_scene(
        target_c=[0.0, 0.0, 1.0], target_half=[0.1, 0.1, 0.1],
        occ_c=[0.05, 0.0, 0.5], occ_half=[0.05, 0.1, 0.1],
        fx=1000.0, fy=1000.0, cx_px=0.0, cy_px=0.0,
    )
    assert hand_calc_visibility(scene) == pytest.approx(0.5, abs=1e-6)


def test_project_aabb_refuses_behind_camera():
    with pytest.raises(ValueError, match="behind the camera"):
        project_aabb_to_rect(
            np.array([0.0, 0.0, -0.5]), np.array([0.1, 0.1, 0.1]), 600, 600, 320, 240,
        )


def test_rect_overlap_zero_when_disjoint():
    rect_a = (0, 10, 0, 10)
    rect_b = (20, 30, 20, 30)
    assert rect_overlap_area(rect_a, rect_b) == 0.0


# --- (3) Template-refusal guard -----------------------------------------------

def test_refuses_on_freshly_written_template():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "validation_scenes.csv")
        write_template_if_missing(path)
        with pytest.raises(ValueError, match="template"):
            refuse_if_still_template(path)


def test_refuses_real_committed_validation_scenes_csv():
    """The actual repo file, as it stands right now, must be refused."""
    real_path = str(PROJECT_ROOT / "perception-calibration-gap" / "validation_scenes.csv")
    with pytest.raises(ValueError):
        refuse_if_still_template(real_path)


def test_passes_once_notes_are_filled_in_for_real():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "validation_scenes.csv")
        write_template_if_missing(path)
        # Simulate a human filling in real measured data (notes no longer
        # contain the "REPLACE with" marker).
        import csv as csv_mod
        with open(path) as f:
            rows = list(csv_mod.DictReader(f))
        for row in rows:
            row["notes"] = "Measured by hand on 2026-09-21, apple diameter 7.2cm."
        with open(path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        refuse_if_still_template(path)  # should NOT raise
