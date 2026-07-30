"""
prg_edge_error_stub.py

STUB for the collaborator's real compute_prg_edge_error(). Returns a
random float in [0,1] so integration testing can run end-to-end before
her real implementation exists. See SCHEMA.md Section 5.

CRITICAL: nothing computed against this stub is ever a real finding.
"""

import numpy as np


def compute_prg_edge_error(rgb, depth, target_bbox, rng: np.random.Generator = None) -> float:
    """STUB. Real signature/interface TBD with collaborator (SCHEMA.md Section 4)."""
    if rng is None:
        rng = np.random.default_rng()
    return float(rng.uniform(0.0, 1.0))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    val = compute_prg_edge_error(rgb=None, depth=None, target_bbox=None, rng=rng)
    print(f"Stub compute_prg_edge_error() returned: {val:.4f}")
    assert 0.0 <= val <= 1.0
    print("Stub self-test passed.")
