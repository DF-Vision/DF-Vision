"""
tests/test_c4_gap_estimator_scenes.py

C4 (project/closure session): three additional synthetic scenes required
by project_closure/CRITERIA.md's C4 section, beyond the idealized
single-edge step (already covered in test_geometric_prg_error_sanity.py):
a two-object touching scene (with far background present, unlike the
idealized edge), a separated scene with a real (non-trivial) gap, and a
background-only scene. All exercise the FIXED _c1_internals() (the
exterior-only ring, see explore/scripts/geometric_prg_error.py).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "explore" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))

import geometric_prg_error as gpe

IMG_SHAPE = (120, 120)
TARGET_BBOX = (20, 20, 60, 60)   # x0,y0,x1,y1 -- a 40x40 square, margin on all sides
FAR_BACKGROUND_M = 2.0
TARGET_DEPTH_M = 0.4


def _base_scene():
    depth = np.full(IMG_SHAPE, FAR_BACKGROUND_M, dtype=np.float32)
    depth[20:60, 20:60] = TARGET_DEPTH_M  # target block
    return depth


def test_two_object_touching_scene_recovers_near_true_gap():
    """Target adjacent (to its right) to a touching neighbor at a true
    ~1mm gap; far background on the other three sides. A harder case than
    the idealized single-edge test since background contamination (the
    E3-diagnosed, separate mechanism) is present here too -- the LOW
    percentile must still favor the near/touching side over the inflated
    background sides."""
    depth = _base_scene()
    true_gap = 0.001
    depth[20:60, 60:100] = TARGET_DEPTH_M + true_gap  # touching neighbor, right side

    internals = gpe._c1_internals(depth, TARGET_BBOX)
    assert internals is not None
    estimated_gap, sigma_gap, score = internals
    print(f"\nTwo-object touching: estimated_gap={estimated_gap:.5f} (true={true_gap}), "
          f"background={FAR_BACKGROUND_M - TARGET_DEPTH_M}")
    assert abs(estimated_gap - true_gap) < abs(estimated_gap - (FAR_BACKGROUND_M - TARGET_DEPTH_M)), (
        f"estimated_gap={estimated_gap:.4f} is closer to the far-background distance "
        f"({FAR_BACKGROUND_M - TARGET_DEPTH_M:.4f}) than to the true touching gap ({true_gap})"
    )
    assert estimated_gap < 0.01, f"estimated_gap={estimated_gap:.4f} should be small (near {true_gap}), not background-scale"


def test_separated_scene_recovers_real_larger_gap():
    """Target and neighbor at a documented larger true gap (5cm), far
    background elsewhere. estimated_gap should recover a value close to
    the true 5cm gap -- not near-zero, not near the background distance."""
    depth = _base_scene()
    true_gap = 0.05
    depth[20:60, 60:100] = TARGET_DEPTH_M + true_gap  # separated neighbor, right side

    internals = gpe._c1_internals(depth, TARGET_BBOX)
    assert internals is not None
    estimated_gap, sigma_gap, score = internals
    print(f"\nSeparated (true 5cm gap): estimated_gap={estimated_gap:.5f} (true={true_gap})")
    assert estimated_gap == pytest.approx(true_gap, abs=0.01), (
        f"estimated_gap={estimated_gap:.4f} does not recover the true {true_gap}m gap"
    )
    assert estimated_gap < (FAR_BACKGROUND_M - TARGET_DEPTH_M) / 2, (
        "estimated_gap should not be dominated by the far background distance"
    )


def test_background_only_scene_reports_large_gap_and_near_zero_score():
    """No neighbor at all -- only far background on every side. estimated_gap
    should be large (near the background distance) and the final score
    should reflect 'decisively not touching' (near 0, since p_contact is
    near 0 for a large gap relative to CONTACT_THRESHOLD_M)."""
    depth = _base_scene()  # no neighbor block added -- background on all sides

    internals = gpe._c1_internals(depth, TARGET_BBOX)
    assert internals is not None
    estimated_gap, sigma_gap, score = internals
    background_gap = FAR_BACKGROUND_M - TARGET_DEPTH_M
    print(f"\nBackground-only: estimated_gap={estimated_gap:.5f} "
          f"(background_gap={background_gap:.4f}), score={score:.4f}")
    assert estimated_gap > 0.5, f"estimated_gap={estimated_gap:.4f} should be large (background-scale)"
    assert score < 0.05, (
        f"score={score:.4f} should be near 0 ('decisively not touching') for a scene "
        f"with only far background and no neighbor"
    )


def test_legacy_classifier_mode_unchanged_by_c4_fix():
    """Criterion 6: this fix touches ONLY explore/scripts/geometric_prg_error.py
    (the classifier-FREE, off-by-default candidates). The default,
    classifier-based prg_edge_error_mode must be completely unaffected --
    delegated to the existing regression suite rather than duplicated here."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         str(PROJECT_ROOT / "tests" / "test_prg_edge_error_mode_regression.py"), "-q"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, (
        f"legacy classifier-mode regression suite failed after the C4 fix:\n"
        f"{result.stdout}\n{result.stderr}"
    )
