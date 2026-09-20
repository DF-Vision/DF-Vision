#!/usr/bin/env python3
"""
explore/scripts/oracle_geometry.py

Ground-truth geometric "oracle" features, computed DIRECTLY from SDF
object geometry (pose + shape parameters) -- no depth camera, no
ray-cast renderer, no LiveDepthFeatureSource involvement of any kind.
This is deliberately independent of every feature-extraction code path
used elsewhere in this project, so it can serve as an honest ceiling on
what depth-derived features could ever recover.

Method: densely sample points (with outward surface normals) on each
object's true 3D surface in its own local frame, transform to world
frame via the object's authored pose, then compute:
  - true_min_gap: minimum Euclidean distance between any sampled point
    on object A's surface and any sampled point on object B's surface.
  - true_contact_normal_alignment: dot(normal_a, -normal_b) at the pair
    of closest sampled points (1.0 = perfectly flush/anti-parallel
    normals, as in a flat face-to-face touch; lower for edge/corner or
    curved-surface contacts).
  - true_vertical_offset: object_a.xyz[2] - object_b.xyz[2] (authored
    center z, not camera/mask-derived).
  - true_horizontal_overlap: IoU of the two objects' world-frame AABB
    XY projections (reuses extract_relation_features.world_aabb(), which
    is itself pure SDF geometry -- not a simplification introduced here,
    just reused since it already does exactly this honestly).

Sampling density (chosen for ~1mm-scale accuracy on ~10-50cm objects
without being prohibitively slow): 12x12 grid per box face (864 pts for
a full box), 24 angular x 12 height positions on a cylinder's lateral
surface plus a 12-ring x 12-angle grid per end cap, and a 400-point
Fibonacci sphere grid for spheres.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))

import numpy as np
import extract_relation_features as erf


def sample_box(dims, n_per_edge=12):
    sx, sy, sz = dims
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    lin = np.linspace(-1, 1, n_per_edge)
    u, v = np.meshgrid(lin, lin)
    u, v = u.ravel(), v.ravel()

    faces = []  # (points, normal)
    faces.append((np.stack([np.full_like(u, -hx), u * hy, v * hz], axis=1), np.array([-1., 0, 0])))
    faces.append((np.stack([np.full_like(u, hx), u * hy, v * hz], axis=1), np.array([1., 0, 0])))
    faces.append((np.stack([u * hx, np.full_like(u, -hy), v * hz], axis=1), np.array([0, -1., 0])))
    faces.append((np.stack([u * hx, np.full_like(u, hy), v * hz], axis=1), np.array([0, 1., 0])))
    faces.append((np.stack([u * hx, v * hy, np.full_like(u, -hz)], axis=1), np.array([0, 0, -1.])))
    faces.append((np.stack([u * hx, v * hy, np.full_like(u, hz)], axis=1), np.array([0, 0, 1.])))

    points = np.concatenate([f[0] for f in faces], axis=0)
    normals = np.concatenate([np.tile(f[1], (len(f[0]), 1)) for f in faces], axis=0)
    return points, normals


def sample_cylinder(dims, n_theta=24, n_h=12, n_cap_r=12, n_cap_theta=12):
    radius, length = dims
    half_len = length / 2
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    h = np.linspace(-half_len, half_len, n_h)
    TH, H = np.meshgrid(theta, h)
    TH, H = TH.ravel(), H.ravel()
    side_pts = np.stack([radius * np.cos(TH), radius * np.sin(TH), H], axis=1)
    side_normals = np.stack([np.cos(TH), np.sin(TH), np.zeros_like(TH)], axis=1)

    r_lin = np.linspace(0, radius, n_cap_r)
    th_lin = np.linspace(0, 2 * np.pi, n_cap_theta, endpoint=False)
    R, TH2 = np.meshgrid(r_lin, th_lin)
    R, TH2 = R.ravel(), TH2.ravel()
    cap_xy = np.stack([R * np.cos(TH2), R * np.sin(TH2)], axis=1)

    top_pts = np.concatenate([cap_xy, np.full((len(cap_xy), 1), half_len)], axis=1)
    top_normals = np.tile(np.array([0., 0., 1.]), (len(cap_xy), 1))
    bot_pts = np.concatenate([cap_xy, np.full((len(cap_xy), 1), -half_len)], axis=1)
    bot_normals = np.tile(np.array([0., 0., -1.]), (len(cap_xy), 1))

    points = np.concatenate([side_pts, top_pts, bot_pts], axis=0)
    normals = np.concatenate([side_normals, top_normals, bot_normals], axis=0)
    return points, normals


def sample_sphere(dims, n_points=400):
    r = dims[0]
    i = np.arange(n_points)
    phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    normals = np.stack([x, y, z], axis=1)
    points = normals * r
    return points, normals


def sample_surface_world(obj) -> tuple[np.ndarray, np.ndarray]:
    """Returns (points_world (N,3), normals_world (N,3)) for one ObjectGeom."""
    if obj.geom_type == "box":
        pts, normals = sample_box(obj.dims)
    elif obj.geom_type == "cylinder":
        pts, normals = sample_cylinder(obj.dims)
    elif obj.geom_type == "sphere":
        pts, normals = sample_sphere(obj.dims)
    else:
        raise ValueError(f"Unhandled geometry type: {obj.geom_type}")

    R = erf._rotation_matrix(obj.rpy)
    pts_world = obj.xyz + pts @ R.T
    normals_world = normals @ R.T
    return pts_world, normals_world


def oracle_pair_features(obj_a, obj_b) -> dict:
    pts_a, normals_a = sample_surface_world(obj_a)
    pts_b, normals_b = sample_surface_world(obj_b)

    # Pairwise distance matrix -- fine at these sample counts (max ~864x864).
    diff = pts_a[:, None, :] - pts_b[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    i_min, j_min = np.unravel_index(np.argmin(dist), dist.shape)
    true_min_gap = float(dist[i_min, j_min])

    na, nb = normals_a[i_min], normals_b[j_min]
    true_contact_normal_alignment = float(np.dot(na, -nb))

    true_vertical_offset = float(obj_a.xyz[2] - obj_b.xyz[2])

    aabb_a, aabb_b = erf.world_aabb(obj_a), erf.world_aabb(obj_b)
    (ax0, ay0, _), (ax1, ay1, _) = aabb_a
    (bx0, by0, _), (bx1, by1, _) = aabb_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    true_horizontal_overlap = float(inter / union) if union > 0 else 0.0

    return {
        "true_min_gap": true_min_gap,
        "true_contact_normal_alignment": true_contact_normal_alignment,
        "true_vertical_offset": true_vertical_offset,
        "true_horizontal_overlap": true_horizontal_overlap,
    }


ORACLE_FEATURE_COLUMNS = [
    "true_min_gap", "true_contact_normal_alignment",
    "true_vertical_offset", "true_horizontal_overlap",
]
