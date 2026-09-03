"""
prg_edge_error_stub.py

STUB for the collaborator's real compute_prg_edge_error() chain. Returns
the SCHEMA.md Section 3a/3b shape (a list of per-edge dicts, one entry
per target-adjacent pair) with a random prg_edge_error in [0,1], so
integration testing can run end-to-end before her real implementation
exists and so the stub path is shape-compatible with
integration/prg_client.py's real return value. See SCHEMA.md Section 5.

CRITICAL: nothing computed against this stub is ever a real finding.
"""

import numpy as np


def compute_prg_edge_error(rgb, depth, target_bbox, rng: np.random.Generator = None) -> list[dict]:
    """STUB. Returns SCHEMA.md Section 3b shape: list[dict] (single synthetic
    edge here, since this stub has no real object masks to enumerate pairs
    from)."""
    if rng is None:
        rng = np.random.default_rng()
    return [{
        "neighbor_name": None,
        "relation_prediction": None,
        "relation_confidence": None,
        "prg_edge_error": float(rng.uniform(0.0, 1.0)),
        "geometric_feature_vector": None,
        "mask_is_bbox_placeholder": False,
    }]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    edges = compute_prg_edge_error(rgb=None, depth=None, target_bbox=None, rng=rng)
    print(f"Stub compute_prg_edge_error() returned: {edges}")
    assert isinstance(edges, list) and len(edges) == 1
    assert 0.0 <= edges[0]["prg_edge_error"] <= 1.0
    print("Stub self-test passed.")
