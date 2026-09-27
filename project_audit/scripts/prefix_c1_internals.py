"""Scratch-only (not committed): pre-fix _c1_internals, extracted verbatim
from git commit bb6bf51^ (the parent of C4's fix commit), for a paired
comparison against the current post-fix version. Never touches the
committed repo."""
import numpy as np
import cv2
from scipy.stats import norm

THIN_RING_PX = 6
INSET_PX = 5
CONTACT_THRESHOLD_M = 0.02
MIN_SIGMA_M = 0.001


def _bbox_to_mask(shape, bbox):
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(shape[1], x1), min(shape[0], y1)
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask, (x0, y0, x1, y1)


def _c1_internals_prefix(depth, target_bbox):
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

    kernel = np.ones((THIN_RING_PX, THIN_RING_PX), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    ring = dilated & ~eroded  # PRE-FIX: straddling ring

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
