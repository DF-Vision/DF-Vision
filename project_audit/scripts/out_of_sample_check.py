#!/usr/bin/env python3
"""
project_audit/scripts/out_of_sample_check.py

C4 audit evidence generator (read-only). Runs the CURRENT (post-fix) C1
estimator on 3 synthetic scenes NOT in explore/'s original idealized-scene
test set nor its world-based evaluation set:
  1-3. An object pair at 3 different camera distances (0.3m, 1.0m, 3.0m),
       same true 1cm gap, to check whether accuracy depends on absolute
       depth scale (the ring/percentile logic operates on raw depth
       differences, not normalized ones -- a plausible failure mode).
  4. A pair with a partial occluder in front of the target (occluder
     nearer to camera than the target, covering part of the target's
     own bbox interior) -- tests whether the INSET_PX interior sample
     used for inner_depth gets contaminated by occluder pixels.
Exact output quoted in project_audit/C4_AUDIT.md section 4.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "explore" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))

import numpy as np
import geometric_prg_error as gpe

TRUE_GAP_M = 0.01
print("SYNTHETIC scratch audit scenes -- NOT part of explore/'s committed test set.\n")


def make_pair_at_distance(base_depth_m, true_gap_m=TRUE_GAP_M, img_size=120):
    depth = np.full((img_size, img_size), base_depth_m + 5.0, dtype=np.float32)  # far background
    depth[20:60, 20:60] = base_depth_m
    depth[20:60, 60:100] = base_depth_m + true_gap_m
    return depth


print("=== 1-3. Object pair at 3 camera distances (true gap = 1cm each) ===")
for base_depth in [0.3, 1.0, 3.0]:
    depth = make_pair_at_distance(base_depth)
    internals = gpe._c1_internals(depth, (20, 20, 60, 60))
    if internals is None:
        print(f"distance={base_depth}m: estimator returned None (no result)")
        continue
    estimated_gap, sigma_gap, score = internals
    error = abs(estimated_gap - TRUE_GAP_M)
    print(f"distance={base_depth}m: estimated_gap={estimated_gap:.6f}m  true_gap={TRUE_GAP_M}m  "
          f"raw_error={error:.6f}m  score={score:.4f}")


def make_partial_occluder_scene(true_gap_m=TRUE_GAP_M, img_size=120):
    """Target block at 20:60,20:60 depth=0.5; neighbor at 20:60,60:100
    depth=0.5+true_gap; an OCCLUDER strip at depth=0.1 (much nearer than
    target) covering the LEFT HALF of the target's own bbox interior
    (x=20:40, y=30:50), simulating a partial occlusion of the target
    itself by something closer to the camera."""
    depth = np.full((img_size, img_size), 5.0, dtype=np.float32)
    depth[20:60, 20:60] = 0.5           # target
    depth[20:60, 60:100] = 0.5 + true_gap_m   # true neighbor
    depth[30:50, 20:40] = 0.1           # occluder covering part of target's own bbox interior
    return depth


print("\n=== 4. Target partially occluded by a nearer object ===")
depth_occ = make_partial_occluder_scene()
internals_occ = gpe._c1_internals(depth_occ, (20, 20, 60, 60))
if internals_occ is None:
    print("estimator returned None (no result)")
else:
    estimated_gap, sigma_gap, score = internals_occ
    error = abs(estimated_gap - TRUE_GAP_M)
    print(f"estimated_gap={estimated_gap:.6f}m  true_gap={TRUE_GAP_M}m  raw_error={error:.6f}m  score={score:.4f}")
    print("(INSET_PX=5 interior sample for inner_depth: with a 40x40 target bbox, "
          "the occluder at x=20:40,y=30:50 overlaps the inset interior region "
          "x=25:55,y=25:55 -- checking whether inner_depth got contaminated:)")
    inner = depth_occ[25:55, 25:55]
    print(f"  inner_depth median actually used would be over this region: "
          f"median={np.median(inner):.4f}m (target's true surface is 0.5m; "
          f"occluder is 0.1m -- if median is far from 0.5, the occluder DID contaminate it)")
