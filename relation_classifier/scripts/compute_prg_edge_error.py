#!/usr/bin/env python3
"""
compute_prg_edge_error.py
--------------------------
Shared interface required by the integration plan:
    compute_prg_edge_error(rgb, depth, target_bbox) -> float in [0, 1]

First version: confidence-based inconsistency + a geometric sanity
check against the depth boundary. Documented here since this is the
methods-section paragraph you'll need later.
"""

import numpy as np


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