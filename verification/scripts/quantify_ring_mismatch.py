#!/usr/bin/env python3
"""
verification/scripts/quantify_ring_mismatch.py

V1a supporting evidence: quantifies the effect of the ring-definition
mismatch found between the training-time boundary exporter
(scripts/export_masks.py / scripts/export_boundary_mask.py: 5x5 ELLIPSE
erosion) and LiveDepthFeatureSource's internal _ring_from_mask() (3x3
SQUARE erosion), on the same ray-cast synthetic masks used in the prior
session's diagnosis. Does not modify any production file.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))

import numpy as np
import cv2

import extract_relation_features as erf
from validate_live_depth_feature_source import ray_cast_scene, HAND_PICKED_PAIRS


def training_style_ring(mask, ring_width=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width, ring_width))
    m = mask.astype(np.uint8)
    eroded = cv2.erode(m, kernel)
    return (m - eroded).astype(bool)


def live_style_ring(mask, ring_px=3):
    return erf._ring_from_mask(mask, ring_px=ring_px)


def main():
    print("{:<22s} {:<28s} {:<10s} {:<10s} {:<8s} {:<10s} {:<10s}".format(
        "world", "pair", "live_ring", "train_ring", "ratio", "live_bs", "train_bs"))
    for world_name, pairs in HAND_PICKED_PAIRS.items():
        objects, cameras = erf.parse_world_sdf(erf.WORLDS_SDF_DIR / f"{world_name}.sdf")
        camera = cameras[1]
        render = ray_cast_scene(camera, objects)
        depth = render.depth
        gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        for name_a, name_b in pairs:
            if name_a not in objects or name_b not in objects:
                continue
            mask_a = render.owner == name_a
            mask_b = render.owner == name_b
            if not np.any(mask_a) or not np.any(mask_b):
                continue

            live_ring_a, live_ring_b = live_style_ring(mask_a), live_style_ring(mask_b)
            train_ring_a, train_ring_b = training_style_ring(mask_a), training_style_ring(mask_b)

            def iface(ring_a, ring_b):
                return (ring_a | ring_b) & (
                    cv2.dilate(mask_a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                    | cv2.dilate(mask_b.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))

            live_iface = iface(live_ring_a, live_ring_b)
            train_iface = iface(train_ring_a, train_ring_b)

            live_bs = np.mean(grad_mag[live_iface]) if live_iface.sum() > 0 else float("nan")
            train_bs = np.mean(grad_mag[train_iface]) if train_iface.sum() > 0 else float("nan")

            ratio = train_iface.sum() / live_iface.sum() if live_iface.sum() > 0 else float("nan")
            print("{:<22s} {:<28s} {:<10d} {:<10d} {:<8.2f} {:<10.4f} {:<10.4f}".format(
                world_name, f"{name_a}/{name_b}", int(live_iface.sum()), int(train_iface.sum()),
                ratio, live_bs, train_bs))


if __name__ == "__main__":
    main()
