#!/usr/bin/env python3
"""
retrain/scripts/r2b_generate_da_v2_features.py

R2's "if a DA-V2 checkpoint exists, ALSO generate DA-V2-depth features"
instruction, and a direct (if partial) test of the long-standing,
never-resolved H1 hypothesis (trained-on-one-depth-source,
tested-on-another) from the last two sessions' diagnoses.

Scope, stated plainly: this generates REAL DA-V2 depth (via direct,
non-ROS Python inference -- torch is available, rclpy is not, see
retrain/BASELINE.md) from REAL RGB frames captured off a live headless
gz sim instance -- genuinely different from every prior session's
ray-cast-only depth. It reuses the RAY-CAST renderer's ground-truth
OBJECT MASKS (not a real segmenter -- none exists yet, SCHEMA.md sec 5)
for the SAME scene/viewpoint, since building full semantic-segmentation
alignment for DA-V2's output resolution was judged out of scope for this
already-large session. This is a clearly-labeled compromise: real depth
source, synthetic (but geometrically correct) masks. Every row is tagged
`depth_source=da_v2_relative` and the DA-V2 model used is the NON-METRIC
"relative" checkpoint (depth_anything_v2_vits.pth) -- output is relative,
unitless depth, NOT metric Z-depth (see retrain/BASELINE.md's caveat).
Metric-scale features (relative_height, minimum_boundary_distance when
is_metric=True) are on an arbitrary scale here and should not be compared
numerically to the ray-cast metric features -- only Sobel-gradient-based
features (boundary_sharpness, depth_gradient) and mask-only features
(overlap_ratio, edge_continuity) are meaningfully comparable.

Bounded scope: 2 worlds x 2 viewpoints (not all 6x6 -- gz sim + RGB
capture + DA-V2 inference per frame has real wall-clock cost neither an
overnight session's budget nor this one's remaining time can spend on a
secondary/nice-to-have column set at full scale).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))

import numpy as np
import pandas as pd
import cv2
import torch

import extract_relation_features as erf
from validate_live_depth_feature_source import ray_cast_scene
from classify_relation import classify_relation

CHECKPOINT = "/home/manasi/bps-project/checkpoints/depth_anything_v2_vits.pth"
OUT_CSV = PROJECT_ROOT / "retrain" / "data" / "relation_dataset_v2_DAV2_SIM.csv"
GC_DIR = PROJECT_ROOT / "verification" / "gazebo_contacts_SIM"

TARGETS = [
    ("world1_baseline", 1), ("world1_baseline", 2),
    ("world4_support_scene", 1), ("world4_support_scene", 2),
]

FEATURE_COLUMNS = [
    "boundary_sharpness", "minimum_boundary_distance", "surface_normal_consistency",
    "relative_height", "depth_gradient", "overlap_ratio", "edge_continuity",
    "occlusion_boundary_score",
]


def load_da_v2_model():
    sys.path.insert(0, str(PROJECT_ROOT))
    from models.depth_anything_v2.dpt import DepthAnythingV2
    cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device).eval()
    print(f"Loaded DA-V2 (relative, vits) on {device}")
    return model


def capture_rgb(world_sdf_path, world_name, viewpoint, timeout=40):
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image

    proc = subprocess.Popen(["gz", "sim", "-s", str(world_sdf_path)])
    try:
        topic = "/rgb_camera" if viewpoint == 1 else f"/rgb_camera_v{viewpoint}"
        node = Node()
        captured = {}

        def cb(msg):
            if "rgb" in captured:
                return
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            captured["rgb"] = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        node.subscribe(Image, topic, cb)
        start = time.time()
        while "rgb" not in captured and time.time() - start < timeout:
            time.sleep(0.1)
        if "rgb" not in captured:
            print(f"[FAIL] {world_name} vp={viewpoint}: no RGB on {topic} within {timeout}s")
            return None
        return captured["rgb"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    if not Path(CHECKPOINT).exists():
        print(f"ERROR: checkpoint not found at {CHECKPOINT} -- skipping DA-V2 generation, "
              f"logging as blocker per rule 10.")
        return

    da_v2 = load_da_v2_model()
    rows = []

    for world_name, vp in TARGETS:
        sdf_path = erf.WORLDS_SDF_DIR / f"{world_name}.sdf"
        objects, cameras = erf.parse_world_sdf(sdf_path)
        if vp not in cameras:
            print(f"[SKIP] {world_name} vp={vp}: no camera")
            continue
        camera = cameras[vp]
        render = ray_cast_scene(camera, objects)  # ground-truth masks come from here

        print(f"\n{world_name} vp={vp}: capturing real RGB from headless gz sim...")
        rgb = capture_rgb(sdf_path, world_name, vp)
        if rgb is None:
            continue

        da_depth = da_v2.infer_image(rgb).astype(np.float32)
        print(f"{world_name} vp={vp}: DA-V2 relative depth shape={da_depth.shape}, "
              f"range=[{da_depth.min():.3f},{da_depth.max():.3f}]")
        if da_depth.shape != render.depth.shape:
            da_depth = cv2.resize(da_depth, (render.depth.shape[1], render.depth.shape[0]))

        gc_path = GC_DIR / f"{world_name}_gazebo_contacts_SIM.csv"
        gc_df = pd.read_csv(gc_path)
        gc_lookup = {frozenset((r.object_a, r.object_b)): r.contact for r in gc_df.itertuples()}

        masks = {name: (render.owner == name) for name in objects if np.any(render.owner == name)}
        pairs = erf.enumerate_adjacent_pairs(masks)
        live = erf.LiveDepthFeatureSource(camera=camera)

        for name_a, name_b in pairs:
            gt = gc_lookup.get(frozenset((name_a, name_b)))
            if gt is None or gt == "unknown":
                continue
            mask_a, mask_b = masks[name_a], masks[name_b]
            feats_da = live.compute(da_depth, mask_a, mask_b)
            feats_raycast = live.compute(render.depth, mask_a, mask_b)
            if any(isinstance(v, float) and np.isnan(v) for v in feats_da.values()):
                continue

            relation_da, conf_da = classify_relation(feats_da)
            row = {
                "world_id": world_name, "viewpoint": vp, "object_a": name_a, "object_b": name_b,
                "gazebo_contact": int(gt), "depth_source": "da_v2_relative",
                "pred_relation_da_v2": relation_da, "pred_confidence_da_v2": conf_da,
            }
            for c in FEATURE_COLUMNS:
                row[f"{c}_da_v2"] = feats_da[c]
                row[f"{c}_raycast"] = feats_raycast[c]
            rows.append(row)

    if not rows:
        print("\nNo rows generated -- DA-V2 secondary feature set is empty. "
              "Logging as a blocker, not fabricating data.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows -> {OUT_CSV}")
    print(df.to_string(index=False))

    print("\n--- Sobel-based feature comparison (only ones meaningfully comparable "
          "across relative vs metric depth) ---")
    for c in ["boundary_sharpness", "depth_gradient"]:
        print(f"{c}: da_v2 mean={df[c+'_da_v2'].mean():.4f}, "
              f"raycast mean={df[c+'_raycast'].mean():.4f}")


if __name__ == "__main__":
    main()
