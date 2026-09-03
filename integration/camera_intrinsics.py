"""
camera_intrinsics.py

Shared CameraIntrinsics type, used by both:
  - integration/prg_client.py (passes this to collaborator's LiveDepthFeatureSource)
  - anywhere on this pipeline's side that has real calibrated intrinsics
    (Orbbec Gemini 2, currently only real_camera/capture_and_gap.py)

Lives here (integration/, not either person's own module) per SCHEMA.md
Section 4 open item 3: both LiveDepthFeatureSource and prg_client.py need
it, and it's an integration-layer type, not specific to either pipeline.
Collaborator's own suggestion was "a shared file rather than either
individual module" -- this is that file. Confirm with her this is an
acceptable home; if she'd rather it live in her repo instead, this file
can be deleted and the import in prg_client.py updated to point at hers --
no other code depends on WHERE this class lives, only on its shape.

Field/property shape mirrors collaborator's own CameraGeom (in
extract_relation_features.py) exactly, EXCEPT fx/fy/cx/cy are stored
fields here, not computed properties derived from horizontal_fov --
confirmed safe by collaborator (2026-07-30): nothing downstream
(depth_to_world_points, feat_minimum_boundary_distance,
feat_boundary_sharpness) reads horizontal_fov directly, only the four
derived attributes themselves. This lets real calibrated intrinsics
(which aren't necessarily square-pixel, unlike an SDF-derived fov) be
passed in directly with no lossy round-trip.

Duck-typing carries this: anything that accepts a CameraGeom-like object
and only reads .fx/.fy/.cx/.cy/.xyz/.rpy/.width/.height/.clip_near/.clip_far
will accept this class without modification.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CameraIntrinsics:
    name: str
    xyz: np.ndarray
    rpy: np.ndarray
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    clip_near: float
    clip_far: float