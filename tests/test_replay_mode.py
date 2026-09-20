"""
tests/test_replay_mode.py

C2: --replay mode for capture_and_gap.py. Exercises the fixture generator
+ replay reader + capture_and_gap's run_replay() end to end, no camera,
no pyorbbecsdk import needed. See project_closure/CRITERIA.md's C2
section for the pre-committed acceptance criteria this test suite checks.
"""
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_CAMERA_DIR = PROJECT_ROOT / "perception-calibration-gap" / "real_camera"
sys.path.insert(0, str(REAL_CAMERA_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "gazebo"))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "mock-pipeline"))
sys.path.insert(0, str(PROJECT_ROOT))

import depth_io
from replay_source import ReplayFrameSource, ReplayFrameError, write_replay_trial, write_manifest
import generate_replay_fixture as fixture_gen
import capture_and_gap
from logger import TrialLogger, REQUIRED_FIELDS


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("replay_fixture_SYNTHETIC")
    fixture_gen.generate(str(d), seed=fixture_gen.RANDOM_STATE)
    return d


# --- depth_io: the ONE conversion point + guard (criterion 3) ---

def test_mm_array_to_meters_conversion():
    mm = np.array([[1000, 500], [0, 2500]], dtype=np.uint16)
    m = depth_io.mm_array_to_meters(mm)
    assert m.dtype == np.float32
    np.testing.assert_allclose(m, [[1.0, 0.5], [0.0, 2.5]])


def test_mm_png_round_trip(tmp_path):
    import cv2
    mm = np.full((10, 10), 750, dtype=np.uint16)
    path = tmp_path / "depth_mm.png"
    cv2.imwrite(str(path), mm)
    meters = depth_io.mm_png_to_meters(str(path))
    assert meters.dtype == np.float32
    np.testing.assert_allclose(meters, 0.75)


def test_validate_depth_meters_rejects_implausible_median():
    huge = np.full((10, 10), 50.0, dtype=np.float32)  # 50m median
    with pytest.raises(depth_io.ImplausibleDepthError, match="median=50"):
        depth_io.validate_depth_meters(huge)


def test_validate_depth_meters_accepts_plausible_scene():
    plausible = np.full((10, 10), 0.5, dtype=np.float32)
    depth_io.validate_depth_meters(plausible)  # must not raise


def test_validate_depth_meters_rejects_all_invalid():
    zeros = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(depth_io.ImplausibleDepthError, match="zero valid"):
        depth_io.validate_depth_meters(zeros)


def test_read_depth_mm_png_rejects_wrong_dtype(tmp_path):
    import cv2
    color_as_depth = np.zeros((10, 10, 3), dtype=np.uint8)
    path = tmp_path / "bad.png"
    cv2.imwrite(str(path), color_as_depth)
    with pytest.raises(RuntimeError, match="expected uint16"):
        depth_io.read_depth_mm_png(str(path))


# --- fixture generator + replay reader share one documented layout ---

def test_fixture_has_8_trials_4_good_4_faulted(fixture_dir):
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    assert len(manifest["trials"]) == 8
    faults = []
    for spec in manifest["trials"]:
        meta = json.loads((fixture_dir / spec["subdir"] / "meta.json").read_text())
        faults.append(meta["inject_fault"])
    assert faults.count(None) == 4
    assert set(f for f in faults if f) == {"missing_depth", "corrupt_depth",
                                            "implausible_depth", "no_marker"}


def test_replay_frame_source_loads_good_trials(fixture_dir):
    source = ReplayFrameSource(str(fixture_dir))
    assert len(source) == 8
    good = [t for t in source if t["trial_id"] < 4]
    for spec in good:
        color, depth_mm, intr = source.load_frame(spec)
        assert color.ndim == 3
        assert depth_mm.dtype == np.uint16
        assert set(intr.keys()) == {"fx", "fy", "cx", "cy"}


def test_replay_frame_source_raises_on_missing_depth(fixture_dir):
    source = ReplayFrameSource(str(fixture_dir))
    spec = next(t for t in source if t["trial_id"] == 4)  # missing_depth
    with pytest.raises(ReplayFrameError, match="Missing depth_mm.png"):
        source.load_frame(spec)


def test_replay_frame_source_raises_on_corrupt_depth(fixture_dir):
    source = ReplayFrameSource(str(fixture_dir))
    spec = next(t for t in source if t["trial_id"] == 5)  # corrupt_depth
    with pytest.raises(ReplayFrameError):
        source.load_frame(spec)


# --- end-to-end: schema-correct rows, determinism, fault handling ---

def _run_replay_to_dict_list(fixture_dir, out_path):
    if out_path.exists():
        out_path.unlink()
    results = capture_and_gap.run_replay(str(fixture_dir), out_path)
    logger = TrialLogger(str(out_path))
    return logger.load_trials()


def test_replay_end_to_end_schema_correct_and_fault_handling(fixture_dir, tmp_path):
    out_path = tmp_path / "replay_out_SYNTHETIC.jsonl"
    trials = _run_replay_to_dict_list(fixture_dir, out_path)

    assert len(trials) == 8, "one row per fixture trial, including faulted ones"

    for t in trials:
        missing_required = [f for f in REQUIRED_FIELDS if t.get(f) is None]
        assert not missing_required, f"trial {t['trial_id']} missing required fields: {missing_required}"

    good_ids = {0, 1, 2, 3}
    faulted_ids = {4, 5, 6, 7}

    for t in trials:
        if t["trial_id"] in good_ids:
            assert t["failure"] == 0 or t["failure"] == 1  # real outcome, not a sentinel
            assert not (isinstance(t["Gap_initial"], float) and math.isnan(t["Gap_initial"])), \
                "good trial should have a real Gap_initial, not NaN"
            assert t["notes"] and "SYNTHETIC REPLAY" in t["notes"]
        else:
            assert t["failure"] == 1, "faulted trial must be flagged as a failure"
            assert math.isnan(t["Gap_initial"]), "faulted trial must use NaN sentinel, not a fabricated Gap"
            assert t["notes"] and ("ERROR" in t["notes"]), \
                f"faulted trial {t['trial_id']} must carry a row-level notes flag, got: {t['notes']}"
            assert t["outcome"] == "input_error"

    # per-stage timing logged for every trial (criterion 8)
    for t in trials:
        assert t.get("stage_timings_ms"), f"trial {t['trial_id']} missing stage_timings_ms"


def test_replay_is_deterministic_across_two_runs(fixture_dir, tmp_path):
    out1 = tmp_path / "run1_SYNTHETIC.jsonl"
    out2 = tmp_path / "run2_SYNTHETIC.jsonl"
    trials1 = _run_replay_to_dict_list(fixture_dir, out1)
    trials2 = _run_replay_to_dict_list(fixture_dir, out2)

    assert len(trials1) == len(trials2)
    # Wall-clock fields (timestamp_utc, stage_timings_ms) excluded per
    # CRITERIA.md's C2 addendum -- everything else must match exactly,
    # including NaN==NaN for faulted trials (math.isnan, not `==`).
    excluded = {"timestamp_utc", "stage_timings_ms"}
    for a, b in zip(trials1, trials2):
        for k in a:
            if k in excluded:
                continue
            va, vb = a[k], b[k]
            if isinstance(va, float) and isinstance(vb, float) and math.isnan(va) and math.isnan(vb):
                continue
            assert va == vb, f"trial {a['trial_id']} field {k} differs: {va!r} vs {vb!r}"


def test_replay_process_does_not_raise_on_bad_fixture_dir(fixture_dir):
    """Sanity: run_replay itself completes (returns) rather than propagating
    any per-trial exception out of the run -- the whole point of C2
    criterion 7."""
    out_path = fixture_dir.parent / "sanity_out_SYNTHETIC.jsonl"
    results = capture_and_gap.run_replay(str(fixture_dir), out_path)
    assert len(results) == 8


# --- masks_are_bbox_placeholder verified at every call site (criterion 9) ---

def _find_prg_client_calls_missing_true_flag(pyfile: Path):
    """AST-based: find every call whose func resolves to something ending in
    get_prg_edge_error, and check it passes masks_are_bbox_placeholder=True
    (as a keyword, literal True). Returns a list of (lineno, reason) for any
    call that doesn't unambiguously pass True."""
    problems = []
    try:
        tree = ast.parse(pyfile.read_text())
    except SyntaxError:
        return problems
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "get_prg_edge_error":
            continue
        flag_kw = next((kw for kw in node.keywords if kw.arg == "masks_are_bbox_placeholder"), None)
        if flag_kw is None:
            problems.append((node.lineno, "masks_are_bbox_placeholder not passed"))
        elif not (isinstance(flag_kw.value, ast.Constant) and flag_kw.value.value is True):
            problems.append((node.lineno, "masks_are_bbox_placeholder not a literal True"))
    return problems


def test_masks_are_bbox_placeholder_true_at_every_call_site():
    py_files = [p for p in PROJECT_ROOT.rglob("*.py")
                if "__pycache__" not in p.parts and ".git" not in p.parts]
    all_problems = {}
    for pf in py_files:
        problems = _find_prg_client_calls_missing_true_flag(pf)
        if problems:
            all_problems[str(pf.relative_to(PROJECT_ROOT))] = problems
    assert all_problems == {}, (
        f"Found get_prg_edge_error() call site(s) not passing "
        f"masks_are_bbox_placeholder=True (neither live nor replay capture "
        f"in this repo has a real segmentation model): {all_problems}"
    )
    # Also confirm at least one real call site exists, so this test isn't
    # vacuously passing because nothing calls get_prg_edge_error at all.
    capture_and_gap_problems_and_calls = [
        pf for pf in py_files
        if "get_prg_edge_error(" in pf.read_text() and "capture_and_gap.py" in pf.name
    ]
    assert capture_and_gap_problems_and_calls, "expected capture_and_gap.py to call get_prg_edge_error"
