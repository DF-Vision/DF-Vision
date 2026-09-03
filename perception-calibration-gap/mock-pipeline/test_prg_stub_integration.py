"""
test_prg_stub_integration.py

Day 5 integration check: confirms the REAL compute_prg_edge_error()
correctly wires into TrialRecord + TrialLogger's real schema, end to end.

Originally tested a local scalar-returning stub; that stub and its
TrialRecord call went stale when logger.py's Rev3 schema replaced the
scalar prg_edge_error field with prg_edges (list[dict], SCHEMA.md Section
3a/3b -- see commit 71543d7). Updated to call the collaborator's real
compute_prg_edge_error() (relation_classifier/scripts/compute_prg_edge_error.py)
directly on synthetic-but-real mask/depth arrays, since prg_client.py's own
three-stage chain is gated behind prg_config.json's use_stub flag and
nothing else in the repo unit-tests this function's real branches in
isolation.

This is NOT a real dual-signal result -- the masks/depth here are synthetic
test fixtures, not a real scene. See SCHEMA.md Section 5.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root, for relation_classifier/

from logger import TrialRecord, TrialLogger
from relation_classifier.scripts.compute_prg_edge_error import compute_prg_edge_error


def main():
    test_log_path = "data/test_prg_stub.jsonl"
    if os.path.exists(test_log_path):
        os.remove(test_log_path)
    freeze_marker = test_log_path + ".FROZEN"
    if os.path.exists(freeze_marker):
        os.remove(freeze_marker)

    logger = TrialLogger(test_log_path)

    # Synthetic depth plane + two masks overlapping at row 5, so the real
    # function's primary interface-overlap branch runs (not the
    # binary_dilation fallback, not the neutral geometric_term=0.5 default).
    depth = np.full((10, 10), 0.5, dtype=np.float32)
    mask_a = np.zeros((10, 10), dtype=bool)
    mask_a[2:6, 2:6] = True
    mask_b = np.zeros((10, 10), dtype=bool)
    mask_b[5:9, 2:6] = True
    relation = "Contact"
    confidence = 0.82

    edge_error = compute_prg_edge_error(
        rgb=None, depth=depth, target_bbox=None,
        relation=relation, confidence=confidence,
        mask_a=mask_a, mask_b=mask_b,
    )
    print(f"Real compute_prg_edge_error() returned: {edge_error:.4f}")
    assert 0.0 <= edge_error <= 1.0, "prg_edge_error out of [0,1] range"

    # SCHEMA.md Section 3b shape: list of per-edge dicts, six fields exactly.
    prg_edges = [{
        "neighbor_name": "synthetic_neighbor",
        "relation_prediction": relation,
        "relation_confidence": confidence,
        "prg_edge_error": edge_error,
        "geometric_feature_vector": None,  # this test calls compute_prg_edge_error directly,
                                            # not the full LiveDepthFeatureSource chain
        "mask_is_bbox_placeholder": True,  # synthetic rectangle test masks, not real segmentation
    }]

    record = TrialRecord(
        trial_id=0, session_id=0, object_id="apple",
        session_baseline_failure_rate=0.05, measured_true_visibility=0.7,
        true_occlusion=0.3, clutter_count=2, perceived_visibility_initial=0.75,
        Gap_initial=0.05, Gap_max=0.08, Gap_slope=0.01, failure=0,
        camera_pose_variant="front_30deg",
        prg_edges=prg_edges,
        prg_is_stub=False,
        dominant_deferral_signal="gap",  # placeholder value, stub-era only
    )
    logger.append_trial(record)
    print("Appended trial with real prg_edges OK")

    loaded = logger.load_trials()
    assert len(loaded) == 1
    assert loaded[0]["prg_edges"][0]["prg_edge_error"] == edge_error, \
        "prg_edge_error did not round-trip correctly"
    assert loaded[0]["dominant_deferral_signal"] == "gap"

    print(f"Round-trip confirmed: prg_edges={loaded[0]['prg_edges']}, "
          f"dominant_deferral_signal={loaded[0]['dominant_deferral_signal']!r}")
    print("\nSTUB INTEGRATION TEST PASSED.")
    print("Reminder: this uses synthetic test masks/depth, never a real scene. "
          "Do not report or analyze this data as if it were dual-signal evidence.")


if __name__ == "__main__":
    main()
