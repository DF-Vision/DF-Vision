#!/usr/bin/env python3
"""
explore/scripts/geometric_prg_error.py

Classifier-free geometric candidates for compute_prg_edge_error(), per
the ORIGINAL, simpler interface stated in
relation_classifier/scripts/compute_prg_edge_error.py's own docstring
("Shared interface required by the integration plan:
compute_prg_edge_error(rgb, depth, target_bbox) -> float in [0, 1]") --
NOT the current, classifier-coupled 7-argument implementation. These
candidates take ONLY (rgb, depth, target_bbox): no relation, no
confidence, no mask_a/mask_b, no call into classify_relation.py or any
detector code. Structural independence (rule/E2): imports only
relation_classifier.scripts.extract_relation_features's pure-geometry
helpers (_ring_from_mask, feat_boundary_sharpness -- reused for genuine
train/serve parity, not reimplemented) plus numpy/cv2/scipy. Verified by
explore/scripts/check_structural_independence.py.

Every candidate treats target_bbox as a coarse rectangle (x0, y0, x1, y1)
in pixel coordinates and looks only at the DEPTH STRUCTURE at and around
that rectangle's own boundary -- it has no notion of a specific
"neighbor object," which is the point: this is meant to work with only
what the shared interface (SCHEMA.md's real contract) actually promises
to supply.

All three return a float in [0,1], polarity matching the project-wide
convention (0 = reliable/consistent, 1 = unreliable/ambiguous) -- same
direction as Gap and the existing compute_prg_edge_error().

--------------------------------------------------------------------------
C1 -- gap-uncertainty
--------------------------------------------------------------------------
Estimates the minimum depth gap between the target and whatever is
immediately outside its bbox (a proxy for "nearest neighbor," since no
neighbor mask is available under this interface), then converts the
gap's own local noise level into an uncertainty band and scores how close
that band sits to a fixed contact threshold.

  ring = pixels in a THIN_RING_PX-wide band centered on the bbox border
         (half inside, half outside -- straddling wherever a real contact
         boundary would actually be).
  inner_depth = median(depth) over the bbox interior, INSET_PX in from
                every edge (avoids the ring itself contaminating the
                "clean surface" estimate).
  For each ring pixel: local_gap = |depth[pixel] - inner_depth|.
  estimated_gap = the 10th percentile of local_gap over the ring (a
                  robust "closest candidate approach distance" --
                  minimum would be a single-pixel outlier, mean would be
                  dominated by far-background pixels; page 10 is a
                  documented, fixed choice, not tuned on results).
  sigma = std(depth) over the SAME small window (INSET_PX square) used
          for inner_depth -- a local noise-level proxy; propagated
          through the gap difference as sigma_gap = sigma * sqrt(2)
          (variance of the difference of two independent, ~equally noisy
          quantities).
  z = (CONTACT_THRESHOLD_M - estimated_gap) / max(sigma_gap, MIN_SIGMA_M)
  p_contact = Phi(z)  (standard normal CDF -- probability true gap is
              below the contact threshold, given the noise estimate)
  C1 = 2 * min(p_contact, 1 - p_contact)
       (0 when p_contact is near 0 or 1 -- decisively touching or not;
       1 when p_contact = 0.5 -- maximally ambiguous given the noise
       level. This is the ONLY formula tested; not adjusted after seeing
       results.)

--------------------------------------------------------------------------
C2 -- boundary sharpness/ambiguity
--------------------------------------------------------------------------
Reuses extract_relation_features.feat_boundary_sharpness() DIRECTLY
(train/serve parity: this is the exact production Sobel-boundary-mean
function, not a reimplementation), applied to the target's own
silhouette ring against itself (ring_b=ring_a, mask_b=mask_a -- the
interface-region formula this function uses reduces cleanly to "the
target's own boundary ring" when both masks are identical, since
ring_a & dilate(mask_a) == ring_a). Raw output is an unbounded Sobel
gradient magnitude mean; C2 = 1 - clip((bs - LO) / (HI - LO), 0, 1)
where LO/HI are the 5th/95th percentiles of bs across the fitting worlds
(non-held-out only, fit once, frozen before touching held-out data) --
LOW gradient magnitude (a smooth, blurry, or occluded boundary) means
HIGH ambiguity (C2 near 1); a SHARP boundary means low ambiguity/error
(C2 near 0).

--------------------------------------------------------------------------
C3 -- fixed combination
--------------------------------------------------------------------------
C3 = max(C1, C2). Chosen a priori (before running either candidate) as
the more conservative combination for a signal meant to gate a physical
manipulation decision: flag as unreliable if EITHER the gap-uncertainty
OR the boundary-sharpness signal indicates trouble, rather than requiring
both (mean) to agree.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))

import numpy as np
from scipy.stats import norm

import extract_relation_features as erf  # ONLY pure-geometry helpers used, see module docstring

THIN_RING_PX = 6
INSET_PX = 5
CONTACT_THRESHOLD_M = 0.02  # matches compute_prg_edge_error.py's existing 2cm convention
MIN_SIGMA_M = 0.001  # floor so a perfectly flat/noiseless synthetic patch doesn't divide by ~0


def _bbox_to_mask(shape, bbox):
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(shape[1], x1), min(shape[0], y1)
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask, (x0, y0, x1, y1)


def _c1_internals(depth, target_bbox):
    """Returns (estimated_gap, sigma_gap, score) or None if the bbox/depth
    don't support a stable estimate. Split out from geometric_c1_gap_uncertainty
    purely so evaluation code (explore/scripts/e3_evaluate_candidates.py)
    can inspect the raw gap estimate for criterion (c) -- the PUBLIC
    geometric_c1_gap_uncertainty() below returns only the final score, its
    behavior is unchanged by this refactor."""
    if depth is None:
        return None
    mask, (x0, y0, x1, y1) = _bbox_to_mask(depth.shape, target_bbox)
    if x1 - x0 < 2 * INSET_PX + 1 or y1 - y0 < 2 * INSET_PX + 1:
        return None

    inner = depth[y0 + INSET_PX:y1 - INSET_PX, x0 + INSET_PX:x1 - INSET_PX]
    inner = inner[np.isfinite(inner)]
    if inner.size == 0:
        return None
    inner_depth = float(np.median(inner))
    sigma = float(np.std(inner)) if inner.size > 1 else MIN_SIGMA_M

    import cv2
    kernel = np.ones((THIN_RING_PX, THIN_RING_PX), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    ring = dilated & ~eroded

    ring_depth = depth[ring]
    ring_depth = ring_depth[np.isfinite(ring_depth)]
    if ring_depth.size == 0:
        return None

    local_gap = np.abs(ring_depth - inner_depth)
    estimated_gap = float(np.percentile(local_gap, 10))
    sigma_gap = max(sigma * np.sqrt(2), MIN_SIGMA_M)

    z = (CONTACT_THRESHOLD_M - estimated_gap) / sigma_gap
    p_contact = float(norm.cdf(z))
    score = float(2 * min(p_contact, 1 - p_contact))
    return estimated_gap, sigma_gap, score


def geometric_c1_gap_uncertainty(rgb, depth, target_bbox) -> float:
    result = _c1_internals(depth, target_bbox)
    return 0.5 if result is None else result[2]


# Fit once on non-held-out worlds, frozen before evaluating held-out data.
# Populated by explore/scripts/e3_evaluate_candidates.py's fit step; a
# reasonable literature/first-principles default is provided so this
# module works standalone, but the E3 script overrides it explicitly and
# logs the fitted values.
_C2_LO_DEFAULT = 0.05
_C2_HI_DEFAULT = 3.0


def _raw_boundary_sharpness(depth, target_bbox):
    """Raw, unnormalized Sobel-mean at the target's own boundary ring, or
    None if unavailable. Exposed for evaluation (criterion (d)'s
    single-feature baseline uses this same raw quantity, min-max scaled)."""
    if depth is None:
        return None
    mask, (x0, y0, x1, y1) = _bbox_to_mask(depth.shape, target_bbox)
    if x1 <= x0 or y1 <= y0:
        return None
    ring = erf._ring_from_mask(mask, ring_px=3)
    if not np.any(ring):
        return None
    bs = erf.feat_boundary_sharpness(depth, ring, ring, mask, mask)
    return float(bs) if np.isfinite(bs) else None


def geometric_c2_boundary_sharpness(rgb, depth, target_bbox, lo: float = _C2_LO_DEFAULT,
                                     hi: float = _C2_HI_DEFAULT) -> float:
    bs = _raw_boundary_sharpness(depth, target_bbox)
    if bs is None:
        return 0.5
    normalized = np.clip((bs - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return float(1.0 - normalized)


def geometric_c3_combined(rgb, depth, target_bbox, lo: float = _C2_LO_DEFAULT,
                           hi: float = _C2_HI_DEFAULT) -> float:
    c1 = geometric_c1_gap_uncertainty(rgb, depth, target_bbox)
    c2 = geometric_c2_boundary_sharpness(rgb, depth, target_bbox, lo=lo, hi=hi)
    return float(max(c1, c2))


CANDIDATES = {
    "C1": geometric_c1_gap_uncertainty,
    "C2": geometric_c2_boundary_sharpness,
    "C3": geometric_c3_combined,
}
