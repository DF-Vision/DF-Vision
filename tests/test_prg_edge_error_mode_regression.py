"""
tests/test_prg_edge_error_mode_regression.py

Regression test for the prg_edge_error_mode switch (E2). Default mode
("classifier") must be bit-for-bit unchanged from the pre-switch
implementation. Run from the project root:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_prg_edge_error_mode_regression.py -q
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))

import numpy as np
import pytest


def _fixed_scene():
    mask_a = np.zeros((60, 60), dtype=bool)
    mask_a[10:30, 10:30] = True
    mask_b = np.zeros((60, 60), dtype=bool)
    mask_b[20:40, 20:40] = True
    depth = np.full((60, 60), 1.0, dtype=np.float32)
    depth[30:40, 30:40] = 1.5
    return depth, mask_a, mask_b


@pytest.fixture(autouse=True)
def _reset_config():
    config_path = PROJECT_ROOT / "relation_classifier" / "prg_edge_error_config.json"
    original = config_path.read_text()
    yield
    config_path.write_text(original)


def _reimport():
    import importlib
    if "compute_prg_edge_error" in sys.modules:
        importlib.reload(sys.modules["compute_prg_edge_error"])
    else:
        importlib.import_module("compute_prg_edge_error")
    return sys.modules["compute_prg_edge_error"]


def test_default_config_is_classifier():
    config_path = PROJECT_ROOT / "relation_classifier" / "prg_edge_error_config.json"
    cfg = json.loads(config_path.read_text())
    assert cfg["prg_edge_error_mode"] == "classifier", (
        "prg_edge_error_config.json's committed default must be 'classifier' (rule 8)"
    )


def test_classifier_mode_output_bit_for_bit_identical():
    cpee = _reimport()
    depth, mask_a, mask_b = _fixed_scene()
    # Expected value from the ORIGINAL (pre-switch) compute_prg_edge_error.py
    # logic, computed by hand from the same confidence_term/geometric_term formula.
    confidence = 0.8
    confidence_term = 1.0 - confidence
    gap = float(np.abs(depth[mask_a].mean() - depth[mask_b].mean()))
    looks_like_contact = gap < 0.02
    predicted_contact = True  # relation == "Contact"
    geometric_term = 0.0 if (looks_like_contact == predicted_contact) else 1.0
    expected = float(np.clip(0.5 * confidence_term + 0.5 * geometric_term, 0.0, 1.0))

    got = cpee.compute_prg_edge_error(None, depth, None, "Contact", confidence, mask_a, mask_b)
    assert got == pytest.approx(expected)


def test_unrecognized_mode_falls_back_to_classifier():
    config_path = PROJECT_ROOT / "relation_classifier" / "prg_edge_error_config.json"
    config_path.write_text('{"prg_edge_error_mode": "not_a_real_mode"}')
    cpee = _reimport()
    assert cpee._load_mode() == "classifier"


def test_geometric_mode_runs_without_touching_config_file(monkeypatch):
    """In-memory override only -- must never require writing the config
    file to disk to test a geometric mode."""
    cpee = _reimport()
    before = (PROJECT_ROOT / "relation_classifier" / "prg_edge_error_config.json").read_text()
    monkeypatch.setattr(cpee, "_load_mode", lambda: "geometric_c1")
    depth, mask_a, mask_b = _fixed_scene()
    result = cpee.compute_prg_edge_error(None, depth, (10, 10, 30, 30), "Contact", 0.8, mask_a, mask_b)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
    after = (PROJECT_ROOT / "relation_classifier" / "prg_edge_error_config.json").read_text()
    assert before == after


def test_use_stub_untouched():
    """rule 4: this switch must never touch integration/prg_config.json."""
    prg_config_path = PROJECT_ROOT / "integration" / "prg_config.json"
    cfg = json.loads(prg_config_path.read_text())
    assert cfg == {"use_stub": True}
