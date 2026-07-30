"""
test_prg_stub_integration.py

Day 5 integration check: confirms the compute_prg_edge_error() STUB
correctly wires into TrialRecord + TrialLogger's real schema, end to end.

This is NOT a real dual-signal result -- the stub returns a random float,
never a real PRG_edge_error computation. See SCHEMA.md Section 5.
"""

import os
import numpy as np
from logger import TrialRecord, TrialLogger


def compute_prg_edge_error(rgb, depth, target_bbox, rng=None) -> float:
    """STUB. Real signature/interface TBD with collaborator (SCHEMA.md Section 4)."""
    if rng is None:
        rng = np.random.default_rng()
    return float(rng.uniform(0.0, 1.0))


def main():
    test_log_path = "data/test_prg_stub.jsonl"
    if os.path.exists(test_log_path):
        os.remove(test_log_path)
    freeze_marker = test_log_path + ".FROZEN"
    if os.path.exists(freeze_marker):
        os.remove(freeze_marker)

    logger = TrialLogger(test_log_path)

    rng = np.random.default_rng(42)
    stub_value = compute_prg_edge_error(rgb=None, depth=None, target_bbox=None, rng=rng)
    print(f"Stub compute_prg_edge_error() returned: {stub_value:.4f}")
    assert 0.0 <= stub_value <= 1.0, "Stub value out of [0,1] range"

    record = TrialRecord(
        trial_id=0, session_id=0, object_id="apple",
        session_baseline_failure_rate=0.05, measured_true_visibility=0.7,
        true_occlusion=0.3, clutter_count=2, perceived_visibility_initial=0.75,
        Gap_initial=0.05, Gap_max=0.08, Gap_slope=0.01, failure=0,
        camera_pose_variant="front_30deg",
        prg_edge_error=stub_value,
        dominant_deferral_signal="gap",  # placeholder value, stub-era only
    )
    logger.append_trial(record)
    print("Appended trial with stub prg_edge_error OK")

    loaded = logger.load_trials()
    assert len(loaded) == 1
    assert loaded[0]["prg_edge_error"] == stub_value, "prg_edge_error did not round-trip correctly"
    assert loaded[0]["dominant_deferral_signal"] == "gap"

    print(f"Round-trip confirmed: prg_edge_error={loaded[0]['prg_edge_error']:.4f}, "
          f"dominant_deferral_signal={loaded[0]['dominant_deferral_signal']!r}")
    print("\nSTUB INTEGRATION TEST PASSED.")
    print("Reminder: this value is a random placeholder, never a real finding. "
          "Do not report or analyze this data as if it were dual-signal evidence.")


if __name__ == "__main__":
    main()
