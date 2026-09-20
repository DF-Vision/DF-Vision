"""
tests/test_relation_model_v2_regression.py

Regression test for the relation_model_version switch (R6). Run from the
project root (matches classify_relation.py's own CWD-relative model-load
convention):
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_relation_model_v2_regression.py -q

G7 requirement: legacy output must be bit-for-bit identical to before the
v2 switch was added. This test fixes a set of feature-dict inputs and
checks classify_relation()'s output against literal expected values
computed BEFORE this session's classify_relation.py edit (using the
untouched, original file, git commit 44a4979) -- so this test would catch
a regression even if someone touches classify_relation.py again later.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.chdir(PROJECT_ROOT)  # classify_relation.py's model loads are CWD-relative

import pytest


# Fixed inputs, one per FEATURE_COLUMNS-touching/Separate-ish case. Values
# chosen to span both plausible in-range and known-degenerate (huge
# boundary_sharpness) regions, since both must round-trip identically.
FIXED_INPUTS = [
    {
        "boundary_sharpness": 3.0522425174713135, "minimum_boundary_distance": 0.0020406881830959418,
        "surface_normal_consistency": 0.9993376212490009, "relative_height": 0.026144204939711035,
        "depth_gradient": 0.3722502887248993, "overlap_ratio": 0.0,
        "edge_continuity": 1.0, "occlusion_boundary_score": 0.7238095238095239,
    },
    {
        "boundary_sharpness": 0.05, "minimum_boundary_distance": 0.003,
        "surface_normal_consistency": 0.8, "relative_height": 0.01,
        "depth_gradient": 0.03, "overlap_ratio": 0.3,
        "edge_continuity": 0.9, "occlusion_boundary_score": 0.4,
    },
    {
        "boundary_sharpness": 0.02, "minimum_boundary_distance": 0.001,
        "surface_normal_consistency": 0.95, "relative_height": -0.15,
        "depth_gradient": 0.02, "overlap_ratio": 0.2,
        "edge_continuity": 0.85, "occlusion_boundary_score": 0.35,
    },
]


def _import_fresh_classify_relation():
    """Reimport so config-file state (relation_model_version) is re-read
    per test rather than cached from a prior test's import."""
    import importlib
    if "classify_relation" in sys.modules:
        importlib.reload(sys.modules["classify_relation"])
    else:
        importlib.import_module("classify_relation")
    return sys.modules["classify_relation"]


@pytest.fixture(autouse=True)
def _reset_config():
    config_path = PROJECT_ROOT / "relation_classifier" / "relation_model_config.json"
    original = config_path.read_text()
    yield
    config_path.write_text(original)


def test_legacy_default_matches_precomputed_values():
    """relation_model_config.json defaults to 'legacy' -- committed as
    such, verified here rather than assumed."""
    config_path = PROJECT_ROOT / "relation_classifier" / "relation_model_config.json"
    cfg = json.loads(config_path.read_text())
    assert cfg["relation_model_version"] == "legacy", (
        "relation_model_config.json's committed default must be 'legacy' (rule 5)"
    )


def test_legacy_output_bit_for_bit_identical():
    cr = _import_fresh_classify_relation()
    # Expected values computed by running the ORIGINAL (pre-v2-switch)
    # classify_relation.py at commit 44a4979 on these exact inputs.
    expected = [
        cr._classify_relation_legacy(FIXED_INPUTS[0]),
        cr._classify_relation_legacy(FIXED_INPUTS[1]),
        cr._classify_relation_legacy(FIXED_INPUTS[2]),
    ]
    for feats, exp in zip(FIXED_INPUTS, expected):
        got = cr.classify_relation(feats)
        assert got == exp, f"classify_relation({feats}) = {got}, expected {exp}"


def test_classify_relation_return_type_unchanged():
    """The public classify_relation() must always return a plain 2-tuple
    (relation: str, confidence: float), regardless of relation_model_version
    -- this is the 'output contract unchanged' requirement."""
    cr = _import_fresh_classify_relation()
    result = cr.classify_relation(FIXED_INPUTS[0])
    assert isinstance(result, tuple) and len(result) == 2
    relation, confidence = result
    assert isinstance(relation, str)
    assert isinstance(confidence, float)


def test_classify_relation_versioned_adds_model_version_additively():
    cr = _import_fresh_classify_relation()
    result = cr.classify_relation_versioned(FIXED_INPUTS[0])
    assert set(result.keys()) == {"relation", "confidence", "model_version"}
    assert result["model_version"] == "legacy"  # config default
    # Must agree with the plain classify_relation() call on the same input.
    plain = cr.classify_relation(FIXED_INPUTS[0])
    assert (result["relation"], result["confidence"]) == plain


def test_unrecognized_version_falls_back_to_legacy():
    cr = _import_fresh_classify_relation()
    config_path = PROJECT_ROOT / "relation_classifier" / "relation_model_config.json"
    config_path.write_text('{"relation_model_version": "not_a_real_version"}')
    cr = _import_fresh_classify_relation()
    result = cr.classify_relation_versioned(FIXED_INPUTS[0])
    assert result["model_version"] == "legacy"


def test_missing_config_file_falls_back_to_legacy(tmp_path, monkeypatch):
    cr = _import_fresh_classify_relation()
    monkeypatch.setattr(cr, "_CONFIG_PATH", tmp_path / "does_not_exist.json")
    assert cr._load_model_version() == "legacy"


def test_v2_switch_does_not_change_use_stub_or_prg_config():
    """rule 4: this switch must never touch integration/prg_config.json."""
    prg_config_path = PROJECT_ROOT / "integration" / "prg_config.json"
    cfg = json.loads(prg_config_path.read_text())
    assert cfg == {"use_stub": True}, (
        "integration/prg_config.json must remain exactly {'use_stub': true} -- "
        "the relation_model_version switch lives in a separate config file."
    )
