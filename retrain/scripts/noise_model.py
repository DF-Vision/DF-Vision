#!/usr/bin/env python3
"""
retrain/scripts/noise_model.py

Depth-noise model for synthetic (ray-cast) training data, re-implemented
fresh this session because no prior noise-model script was ever committed
(same fate as the lost training CSV -- confirmed via repo-wide grep,
nothing found). Documented, literature-grounded approximation, per the
description already on record from an earlier (uncommitted) session:
  1. Edge blending -- smooths hard silhouette transitions over ~N pixels,
     approximating the SDF's own declared <anti_aliasing>4</anti_aliasing>.
  2. Distance-scaled Gaussian noise -- real depth sensors (structured-
     light / ToF) have axial noise that grows with range. Approximate
     form after Nguyen, Izadi & Lovell (2012)'s Kinect-v1 noise model:
       sigma(z) = sigma0 + k * (z - z_ref)^2   [z, z_ref in metres]
     Nguyen et al.'s original constants are calibrated to a specific
     sensor (PrimeSense/Kinect v1) that isn't necessarily representative
     of whatever camera this project's real hardware will use -- treated
     here as a documented, disclosed APPROXIMATION, not a validated match
     to any specific real sensor. sigma0=0.001, k=0.0035, z_ref=0.5 (close
     to Nguyen et al.'s reported axial-noise magnitudes at ~1-2m range).
  3. 1mm quantization -- many depth sensors and depth-image formats quantize
     to millimetre resolution.

`level` is a multiplier applied to steps 1 and 2 uniformly (0 = no noise
at all, matching a noiseless ray-cast render; 1 = the base magnitude
above; up to 15 for last session's stress-test range). Quantization step
size is NOT scaled by level (a real sensor's bit depth doesn't change
with distance) -- always 1mm when level > 0, none when level == 0.
"""
from __future__ import annotations

import numpy as np
import cv2


def apply_depth_noise(depth: np.ndarray, level: float, seed: int,
                       edge_blend_px: int = 4,
                       sigma0: float = 0.001, k: float = 0.0035, z_ref: float = 0.5,
                       quantize_m: float = 0.001) -> np.ndarray:
    """Returns a NEW array; does not modify depth in place.

    level=0 returns depth unchanged (bit-for-bit) -- this is the "clean"
    condition used for the primary training-data generation; level>0 is
    used for the robustness sweep (R4) and for randomized per-scene noise
    injection in R2 (never level=0 alone, so the model isn't fit purely
    on unrealistically clean depth).
    """
    if level <= 0:
        return depth.copy()

    rng = np.random.default_rng(seed)
    d = depth.astype(np.float64).copy()

    # 1. Edge blending: box-blur the whole depth image with a kernel sized
    # to edge_blend_px, then blend it in from the true edges outward using
    # a Sobel-magnitude edge mask so flat interior regions are left alone
    # (matches the physical picture: anti-aliasing only affects silhouette
    # transitions, not interior surface noise, which is handled separately
    # by step 2 below).
    if edge_blend_px > 0:
        ksize = max(1, int(round(edge_blend_px * level)))
        if ksize % 2 == 0:
            ksize += 1
        if ksize > 1:
            blurred = cv2.GaussianBlur(d.astype(np.float32), (ksize, ksize), 0).astype(np.float64)
            gx = cv2.Sobel(d.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(d.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            edge_mag = np.sqrt(gx ** 2 + gy ** 2)
            edge_mask = np.clip(edge_mag / (np.percentile(edge_mag, 99) + 1e-9), 0, 1)
            d = d * (1 - edge_mask) + blurred * edge_mask

    # 2. Distance-scaled Gaussian noise (Nguyen et al. 2012 functional form,
    # disclosed approximation -- see module docstring).
    sigma_map = (sigma0 + k * (d - z_ref) ** 2) * level
    d = d + rng.normal(0.0, 1.0, size=d.shape) * sigma_map

    # 3. Quantization.
    if quantize_m > 0:
        d = np.round(d / quantize_m) * quantize_m

    return d.astype(np.float32)
