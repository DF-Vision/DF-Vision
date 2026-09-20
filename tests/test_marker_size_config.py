"""
tests/test_marker_size_config.py

C7: no test existed for capture_marker_points.py::parse_marker_sizes
(the marker-size config read this project's calibration path depends on
-- a wrong marker size silently corrupts every downstream solvePnP pose,
per PREREG.md's own stated concern). Written before any mutation is
attempted against it, per this session's criteria-before-implementation
discipline.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "gazebo"))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "calibration"))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap"))

from capture_marker_points import parse_marker_sizes


def test_single_pair_cm_to_m():
    assert parse_marker_sizes("0:5.0") == {0: 0.05}


def test_multiple_pairs():
    result = parse_marker_sizes("0:5.0,1:3.5,5:10.0")
    assert result == {0: 0.05, 1: 0.035, 5: 0.10}


def test_cm_to_m_conversion_is_exact_division_by_100():
    result = parse_marker_sizes("2:4.5")
    assert result[2] == pytest.approx(0.045)


def test_marker_id_parsed_as_int_not_string():
    result = parse_marker_sizes("7:6.0")
    assert isinstance(list(result.keys())[0], int)
    assert 7 in result
    assert "7" not in result
