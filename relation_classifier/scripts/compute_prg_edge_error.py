#!/usr/bin/env python3
"""
compute_prg_edge_error.py
--------------------------
Shared interface required by the integration plan:
    compute_prg_edge_error(rgb, depth, target_bbox) -> float in [0, 1]

First version: confidence-based inconsistency + a geometric sanity
check against the depth boundary. Documented here since this is the
methods-section paragraph you'll need later.

MODE SWITCH (added, explore/geometric-prg-error session): this function's
DEFAULT behavior (mode "classifier", see below) is completely unchanged
-- same confidence+geometric-consistency computation, same required
relation/confidence/mask_a/mask_b arguments. An experimental,
classifier-free geometric mode (explore/scripts/geometric_prg_error.py's
C1/C2/C3 candidates) can be selected via
relation_classifier/prg_edge_error_config.json's `prg_edge_error_mode`
key -- NEVER via integration/prg_config.json, whose `use_stub` key is a
separate, untouched concern. Default is "classifier"; the geometric
candidates are OFF unless someone deliberately edits that file. When a
geometric mode is active, relation/confidence/mask_a/mask_b are still
accepted (signature unchanged, for caller compatibility) but IGNORED --
only rgb/depth/target_bbox are used, matching the geometric candidates'
own classifier-free design. See explore/CRITERIA.md and
explore/E3_FINDINGS.md for whether any candidate is actually recommended;
as of that write-up, none has been promoted to a new default.
"""

import json
import sys
from pathlib import Path

import numpy as np

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "prg_edge_error_config.json"
_EXPLORE_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "explore" / "scripts"


def _load_mode() -> str:
    """Never raises -- an absent/malformed config file means 'classifier',
    the safe, unchanged default."""
    try:
        cfg = json.loads(_CONFIG_PATH.read_text())
        mode = cfg.get("prg_edge_error_mode", "classifier")
        if mode not in ("classifier", "geometric_c1", "geometric_c2", "geometric_c3"):
            return "classifier"
        return mode
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return "classifier"


def _geometric_candidates():
    if str(_EXPLORE_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_EXPLORE_SCRIPTS_DIR))
    import geometric_prg_error
    return geometric_prg_error.CANDIDATES


def compute_prg_edge_error(rgb, depth, target_bbox, relation: str, confidence: float,
                            mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Combines:
      (a) classifier confidence — low confidence -> higher error
      (b) geometric consistency — does the predicted relation match a
          simple depth-boundary heuristic (near-zero gap at the
          interface implies Contact; a large gap implies Separate)?

    Returns a scalar in [0, 1]; 0 = fully consistent, 1 = maximally
    inconsistent.
    """
    mode = _load_mode()
    if mode != "classifier":
        candidate_fn = _geometric_candidates()[{"geometric_c1": "C1", "geometric_c2": "C2",
                                                 "geometric_c3": "C3"}[mode]]
        return candidate_fn(rgb, depth, target_bbox)

    confidence_term = 1.0 - float(np.clip(confidence, 0.0, 1.0))

    interface = mask_a.astype(bool) & mask_b.astype(bool)
    if not np.any(interface):
        # Masks don't touch/overlap at all — fall back to dilated boundary gap
        from scipy.ndimage import binary_dilation
        dilated_a = binary_dilation(mask_a.astype(bool), iterations=3)
        interface = dilated_a & mask_b.astype(bool)

    if np.any(interface) and depth is not None:
        gap = float(np.nanmean(np.abs(
            depth[mask_a.astype(bool)].mean() - depth[mask_b.astype(bool)].mean()
        )))
        looks_like_contact = gap < 0.02  # 2cm threshold, tune after smoke-test
        predicted_contact = relation == "Contact"
        geometric_term = 0.0 if (looks_like_contact == predicted_contact) else 1.0
    else:
        geometric_term = 0.5  # insufficient info — neutral, not zero

    return float(np.clip(0.5 * confidence_term + 0.5 * geometric_term, 0.0, 1.0))