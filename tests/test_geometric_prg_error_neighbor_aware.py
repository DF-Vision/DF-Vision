"""
tests/test_geometric_prg_error_neighbor_aware.py

C5: the neighbor-aware PRG candidate (explore/scripts/geometric_prg_error_neighbor_aware.py)
FAILED explore/CRITERIA.md's sensitivity gate (criterion b, 0/5 worlds) --
see project_closure/LOG.md's C5 entry and
explore/scripts/e5_evaluate_neighbor_aware_candidate.py's output. NOT
selected, NOT wired into the default pipeline. This test file exists so
the (genuinely correct, on idealized scenes) neighbor-identification
logic this candidate contributes stays covered and doesn't silently
rot/break, per this project's general testing discipline -- it does NOT
claim the candidate passes the real evaluation criteria.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "explore" / "scripts"))

import geometric_prg_error_neighbor_aware as na


def test_idealized_step_edge_recovers_true_gap():
    depth = np.full((80, 80), 0.5, dtype=np.float32)
    depth[:, 40:] = 0.501
    internals = na._internals(depth, (0, 0, 40, 80))
    assert internals is not None
    estimated_gap, sigma_local, score = internals
    assert estimated_gap == pytest.approx(0.001, abs=0.0005)


def test_two_object_touching_ignores_far_background():
    """The key design goal this candidate was built to test: does NOT
    fall back to the far background just because it surrounds the target
    on more sides than the true (closer, real) neighbor."""
    depth = np.full((120, 120), 2.0, dtype=np.float32)
    depth[20:60, 20:60] = 0.4
    depth[20:60, 60:100] = 0.401
    internals = na._internals(depth, (20, 20, 60, 60))
    assert internals is not None
    estimated_gap, sigma_local, score = internals
    assert estimated_gap == pytest.approx(0.001, abs=0.0005), (
        f"estimated_gap={estimated_gap} should recover the true 1mm neighbor gap, "
        f"not the 1.6m background distance"
    )


def test_separated_scene_recovers_real_gap_not_background():
    depth = np.full((120, 120), 2.0, dtype=np.float32)
    depth[20:60, 20:60] = 0.4
    depth[20:60, 60:100] = 0.45  # 5cm separated neighbor
    internals = na._internals(depth, (20, 20, 60, 60))
    assert internals is not None
    estimated_gap, sigma_local, score = internals
    assert estimated_gap == pytest.approx(0.05, abs=0.005)


def test_background_only_scene_falls_back_to_background():
    """No real neighbor exists -- correctly reports the background
    distance via the documented fallback, not a fabricated small gap."""
    depth = np.full((120, 120), 2.0, dtype=np.float32)
    depth[20:60, 20:60] = 0.4
    internals = na._internals(depth, (20, 20, 60, 60))
    assert internals is not None
    estimated_gap, sigma_local, score = internals
    assert estimated_gap == pytest.approx(1.6, abs=0.01)


def test_score_is_deterministic_and_in_range():
    depth = np.full((120, 120), 2.0, dtype=np.float32)
    depth[20:60, 20:60] = 0.4
    depth[20:60, 60:100] = 0.401
    bbox = (20, 20, 60, 60)
    s1 = na.geometric_neighbor_aware(None, depth, bbox)
    s2 = na.geometric_neighbor_aware(None, depth, bbox)
    assert s1 == s2
    assert 0.0 <= s1 <= 1.0


def test_no_second_region_returns_neutral_score():
    """A perfectly flat depth image (no discontinuities at all -- n_labels<=1)
    must return the documented neutral fallback, not raise."""
    depth = np.full((80, 80), 0.5, dtype=np.float32)
    score = na.geometric_neighbor_aware(None, depth, (10, 10, 30, 30))
    assert score == 0.5
