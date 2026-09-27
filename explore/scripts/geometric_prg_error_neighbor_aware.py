#!/usr/bin/env python3
"""
explore/scripts/geometric_prg_error_neighbor_aware.py

C5 (project/closure session): a changed-interface `compute_prg_edge_error`
candidate, per project_closure/CRITERIA.md's C5 section. Same
`(rgb, depth, target_bbox) -> float in [0,1]` interface as C1/C2/C3
(explore/scripts/geometric_prg_error.py), but fixes the specific
mechanism E3 diagnosed for why those three failed: sampling an
OMNIDIRECTIONAL ring around the target's own bbox border has no
knowledge of which direction a real neighbor lies, so it's dominated by
whatever is closest in ANY direction -- frequently far background, not
the true touching partner.

This candidate instead:
  1. Segments the depth image into connected components using ONLY depth
     discontinuities (Sobel gradient magnitude threshold on depth ->
     smooth-region mask -> cv2.connectedComponents). No YOLO/TSDF/Gap
     import; no ground-truth object labels used here (only by the
     evaluation harness, to score this candidate against the truth).
  2. Identifies the TARGET component: whichever depth-discontinuity
     region overlaps target_bbox's interior the most.
  3. Identifies the NEIGHBOR component: whichever OTHER region shares
     the most boundary-adjacent pixels with the target region (dilate
     the target region by a few px, find which other labels it
     touches, pick the one with the MOST touching pixels -- not the
     first one found). If nothing touches the target at all, falls back
     to the nearest other region by minimum pixel-space boundary
     distance (a "separated, not touching" case).
  4. estimated_gap = the minimum |depth(target boundary pixel) -
     depth(neighbor boundary pixel)| over the touching (or nearest, if
     not touching) boundary pixel pairs SPECIFICALLY between these two
     identified regions -- never an omnidirectional ring.
  5. score = uncertainty of that gap estimate:
         sigma_local = std(local_gap over the boundary pixel pairs used)
         score = sigma_local / (sigma_local + SCALE_M)
     0 when the interface is perfectly consistent (low internal spread,
     high confidence in estimated_gap); approaches 1 as the estimate's
     own internal spread grows relative to SCALE_M. This is the ONLY
     formula tested, fixed before running, not adjusted after seeing
     results (same discipline as C1/C2/C3's own docstring commitment).
"""
from __future__ import annotations

import numpy as np
import cv2

EDGE_GRADIENT_THRESHOLD_M_PER_PX = 0.0015  # depth jump/px flagged as a discontinuity --
                                             # tuned on toy synthetic scenes (not the real
                                             # evaluation worlds) BEFORE running against
                                             # explore/CRITERIA.md: must be small enough to
                                             # separate a 1mm-gap Contact/Support pair (the
                                             # scale explore/E1_FINDINGS.md found matters most)
                                             # without needing per-scene tuning.
MIN_COMPONENT_AREA_PX = 20
DILATE_PX = 3
SCALE_M = 0.02  # matches CONTACT_THRESHOLD_M's existing 2cm convention (geometric_prg_error.py)


def _segment_by_depth_discontinuity(depth: np.ndarray):
    """Returns (labels, n_labels). label 0 = discontinuity/edge pixels
    (not a real region); labels 1..n-1 = smooth depth regions, found
    WITHOUT any object identity information."""
    valid = np.isfinite(depth)
    d = np.where(valid, depth, 0.0).astype(np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_mask = (grad_mag > EDGE_GRADIENT_THRESHOLD_M_PER_PX) | (~valid)
    smooth_mask = (~edge_mask).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(smooth_mask, connectivity=8)
    return labels, n_labels


def _target_label(labels: np.ndarray, target_bbox, inset_px: int = 3):
    x0, y0, x1, y1 = [int(round(v)) for v in target_bbox]
    h, w = labels.shape
    x0i, y0i = max(0, x0 + inset_px), max(0, y0 + inset_px)
    x1i, y1i = min(w, x1 - inset_px), min(h, y1 - inset_px)
    if x1i <= x0i or y1i <= y0i:
        return None
    inner = labels[y0i:y1i, x0i:x1i]
    inner = inner[inner > 0]
    if inner.size == 0:
        return None
    values, counts = np.unique(inner, return_counts=True)
    return int(values[np.argmax(counts)])


def _boundary_pairs_for_label(target_mask, other_mask, depth):
    """Matches each other_mask-in-the-touching-ring pixel to its nearest
    target_mask pixel. Shared by the touching and fallback paths below."""
    from scipy.spatial import cKDTree
    ys, xs = np.where(other_mask)
    if ys.size == 0:
        return None
    target_ys, target_xs = np.where(target_mask)
    if target_ys.size == 0:
        return None
    tree = cKDTree(np.stack([target_ys, target_xs], axis=1))
    _, idx = tree.query(np.stack([ys, xs], axis=1))
    return (target_ys[idx], target_xs[idx]), (ys, xs)


def _neighbor_boundary_pairs(labels: np.ndarray, target_label: int, target_bbox, depth: np.ndarray):
    """Returns (target_pixels_yx, neighbor_pixels_yx) -- equal-length
    arrays of matched boundary pixel coordinates between the target
    region and its identified neighbor region, or None if no other
    region exists at all.

    Neighbor SELECTION among multiple touching candidates: picks the
    touching label whose interface has the SMALLEST median depth
    discontinuity with the target, not whichever has the most touching
    pixels. This matters because a small target is often surrounded by
    background on most of its perimeter and a real touching neighbor on
    only a short segment -- "most touching pixels" would wrongly select
    the background every time (confirmed by direct testing on a
    synthetic two-object scene before this fix). Assuming the true
    touching/nearest neighbor is whichever adjacent region sits closest
    in depth to the target (not simply whichever has more perimeter
    contact) is the key design decision for this candidate, fixed here
    before any evaluation was run against explore/CRITERIA.md."""
    target_mask = (labels == target_label)
    if not np.any(target_mask):
        return None

    kernel = np.ones((2 * DILATE_PX + 1, 2 * DILATE_PX + 1), np.uint8)
    dilated_target = cv2.dilate(target_mask.astype(np.uint8), kernel).astype(bool)
    touching_ring = dilated_target & ~target_mask

    touching_labels_in_ring = labels[touching_ring]
    candidate_labels = np.unique(touching_labels_in_ring[
        (touching_labels_in_ring > 0) & (touching_labels_in_ring != target_label)
    ])

    if candidate_labels.size > 0:
        best_label, best_median_gap, best_pairs = None, None, None
        for cand in candidate_labels:
            cand_ring_mask = touching_ring & (labels == cand)
            pairs = _boundary_pairs_for_label(target_mask, cand_ring_mask, depth)
            if pairs is None:
                continue
            (ty, tx), (ny, nx) = pairs
            gaps = np.abs(depth[ty, tx] - depth[ny, nx])
            gaps = gaps[np.isfinite(gaps)]
            if gaps.size == 0:
                continue
            median_gap = float(np.median(gaps))
            if best_median_gap is None or median_gap < best_median_gap:
                best_label, best_median_gap, best_pairs = cand, median_gap, pairs
        if best_pairs is not None:
            return best_pairs

    # No component touches the target at all (or none had valid depth) --
    # fall back to nearest OTHER region by pixel-space boundary distance.
    other_mask = (labels > 0) & (labels != target_label)
    if not np.any(other_mask):
        return None
    from scipy.spatial import cKDTree
    ys_t, xs_t = np.where(target_mask)
    ys_o, xs_o = np.where(other_mask)
    tree = cKDTree(np.stack([ys_o, xs_o], axis=1))
    dist, idx = tree.query(np.stack([ys_t, xs_t], axis=1))
    k = max(1, min(50, len(dist)))
    closest = np.argsort(dist)[:k]
    return (ys_t[closest], xs_t[closest]), (ys_o[idx[closest]], xs_o[idx[closest]])


def _internals(depth: np.ndarray, target_bbox):
    """Returns (estimated_gap, sigma_local, score) or None."""
    if depth is None:
        return None
    labels, n_labels = _segment_by_depth_discontinuity(depth)
    if n_labels <= 1:
        return None
    target_label = _target_label(labels, target_bbox)
    if target_label is None:
        return None
    pairs = _neighbor_boundary_pairs(labels, target_label, target_bbox, depth)
    if pairs is None:
        return None
    (ty, tx), (ny, nx) = pairs
    target_depths = depth[ty, tx]
    neighbor_depths = depth[ny, nx]
    valid = np.isfinite(target_depths) & np.isfinite(neighbor_depths)
    if valid.sum() == 0:
        return None
    local_gap = np.abs(target_depths[valid] - neighbor_depths[valid])
    estimated_gap = float(np.min(local_gap))
    sigma_local = float(np.std(local_gap)) if local_gap.size > 1 else 0.0
    score = float(sigma_local / (sigma_local + SCALE_M))
    return estimated_gap, sigma_local, score


def geometric_neighbor_aware(rgb, depth, target_bbox) -> float:
    result = _internals(depth, target_bbox)
    return 0.5 if result is None else result[2]


CANDIDATES = {"N1": geometric_neighbor_aware}
