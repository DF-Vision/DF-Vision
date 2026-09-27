"""
tests/test_e5_candidate_harness.py

project/closure-actions G3: verifies the --candidate pluggability added to
explore/scripts/e5_analyze_real_capture_gaps.py. Does NOT run against any
real capture data (none exists) -- builds a small SYNTHETIC capture-dir
fixture (a two-object-touching scene with a known 1mm true gap, same
style as tests/test_c4_gap_estimator_scenes.py) and confirms the harness
is ready for all 4 candidates. No performance claim is made or reported
for c5-neighbor-aware here -- this only proves the harness runs it
without error, per this task's explicit scope.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "explore" / "scripts" / "e5_analyze_real_capture_gaps.py"


@pytest.fixture(scope="module")
def capture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("e5_fixture_SYNTHETIC")
    scene_dir = d / "scene01_test"
    scene_dir.mkdir()

    depth = np.full((120, 120), 2.0, dtype=np.float32)
    depth[20:60, 20:60] = 0.4
    depth[20:60, 60:100] = 0.401  # true gap = 1mm
    np.save(scene_dir / "depth_1.npy", depth)
    np.save(scene_dir / "depth_2.npy", depth)

    rgb = np.full((120, 120, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(scene_dir / "rgb_1.png"), rgb)
    cv2.imwrite(str(scene_dir / "rgb_2.png"), rgb)

    (d / "ground_truth.csv").write_text(
        "scene_folder,object_a,object_b,true_relation,measured_gap_m,measurement_tool,notes\n"
        "scene01_test,apple,purse,Contact,0.001,calipers,SYNTHETIC fixture for e5 harness regression test\n"
    )
    (d / "bbox.csv").write_text(
        "scene_folder,object,x0,y0,x1,y1\n"
        "scene01_test,apple,20,20,60,60\n"
    )
    return d


def _run(capture_dir, candidate=None):
    cmd = [sys.executable, str(SCRIPT), "--capture-dir", str(capture_dir),
           "--bbox-csv", str(capture_dir / "bbox.csv")]
    if candidate is not None:
        cmd += ["--candidate", candidate]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                             env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
    return result


# (a) every candidate, including c5-neighbor-aware, selectable and runs without error
@pytest.mark.parametrize("candidate", ["c1", "c2", "c3", "c5-neighbor-aware"])
def test_each_candidate_selectable_and_runs_without_error(capture_dir, candidate):
    result = _run(capture_dir, candidate=candidate)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    if candidate == "c5-neighbor-aware":
        assert "C5-neighbor-aware" in result.stdout
        assert "C1:" not in result.stdout  # legacy AUROC line must not appear when only c5 selected
    else:
        assert f"{candidate.upper()}:" in result.stdout


# (b) default (all) reports all 4
def test_default_all_reports_all_four_candidates(capture_dir):
    result = _run(capture_dir)  # no --candidate flag at all
    assert result.returncode == 0
    for marker in ["C1:", "C2:", "C3:", "C5-neighbor-aware:"]:
        assert marker in result.stdout, f"missing {marker} in default (all) output:\n{result.stdout}"


# (c) c1/c2/c3 output is unchanged from before this change -- regression
# assertion against the EXACT text the pre-change script (git HEAD before
# this task's commit) produced on this identical fixture, captured once
# by directly running that version -- see project_closure_actions/HANDOFF.md's
# G3 section for the side-by-side proof this golden text came from.
GOLDEN_C1_C2_C3_LINES = [
    "n=2, wrong fraction (error > 1mm) = 0.000",
    "C1: AUROC(score, wrong) = nan (E3's simulation result for comparison: C1=0.457, C2=0.486, C3=0.463)",
    "C2: AUROC(score, wrong) = nan (E3's simulation result for comparison: C1=0.457, C2=0.486, C3=0.463)",
    "C3: AUROC(score, wrong) = nan (E3's simulation result for comparison: C1=0.457, C2=0.486, C3=0.463)",
]


def test_c1_c2_c3_output_unchanged_from_pre_change_golden(capture_dir):
    result = _run(capture_dir)  # default "all"
    assert result.returncode == 0
    for golden_line in GOLDEN_C1_C2_C3_LINES:
        assert golden_line in result.stdout, f"missing/changed golden line: {golden_line!r}"


def test_c1_c2_c3_dataframe_values_match_golden_exactly():
    """Parses the actual per-frame numbers directly (not just the printed
    text) and compares to the exact values the pre-change script produced
    on this identical fixture."""
    sys.path.insert(0, str(PROJECT_ROOT / "explore" / "scripts"))
    import geometric_prg_error as gpe

    depth = np.full((120, 120), 2.0, dtype=np.float32)
    depth[20:60, 20:60] = 0.4
    depth[20:60, 60:100] = 0.401
    bbox = (20, 20, 60, 60)

    internals = gpe._c1_internals(depth, bbox)
    assert internals is not None
    estimated_gap, sigma_gap, c1_score = internals
    c2_score = gpe.geometric_c2_boundary_sharpness(None, depth, bbox)
    c3_score = max(c1_score, c2_score)

    # Exact values the pre-change script produced on this fixture
    # (verified by directly running git HEAD's version before this
    # change was committed -- see HANDOFF.md's G3 section).
    assert estimated_gap == pytest.approx(0.001, abs=1e-6)
    assert c1_score == pytest.approx(0.0, abs=1e-9)
    assert c2_score == pytest.approx(0.0, abs=1e-9)
    assert c3_score == pytest.approx(0.0, abs=1e-9)


# G3's explicit scope limit: no performance number is asserted for
# c5-neighbor-aware anywhere in this file -- only that it runs.
def test_c5_neighbor_aware_produces_no_performance_claim_in_this_test_file():
    """Meta-test: greps this file's own source to confirm no assertion
    anywhere claims a specific AUROC/accuracy number FOR c5-neighbor-aware
    (as opposed to merely checking it ran) -- guards against a future
    editor accidentally turning this into a result claim."""
    own_source = Path(__file__).read_text()
    # the only numeric claims in this file are about C1/C2/C3's own
    # long-established golden values and the shared fixture's true gap;
    # no "c5" line should contain an asserted AUROC/accuracy figure.
    for line in own_source.splitlines():
        if "c5" in line.lower() and ("auroc" in line.lower() or "accuracy" in line.lower()):
            assert "==" not in line and "pytest.approx" not in line, (
                f"found a performance assertion for c5-neighbor-aware, out of this task's scope: {line}"
            )
