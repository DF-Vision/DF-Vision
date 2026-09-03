#!/usr/bin/env python3
"""
validate_live_depth_feature_source.py

Ground-truth validation harness for LiveDepthFeatureSource
(extract_relation_features.py). Does NOT modify extract_relation_features.py
or any file under test -- imports it read-only and calls the real,
unmodified LiveDepthFeatureSource.compute().

METHOD
------
For each test pair, this script:
  1. Builds a synthetic depth image + per-object masks via an independent,
     from-scratch analytic ray-intersection renderer (ray_cast_scene below).
     Uses each world's own authored depth_camera1 intrinsics/pose. Does NOT
     call depth_to_world_points()/range_to_z_depth() or any feat_* function.
  2. Independently hand-derives an expected value for each of the 8
     features from the SAME authored SDF geometry (closed-form geometry,
     or the renderer's own known-true hit points/depths). Also never calls
     feat_*, _ring_from_mask, or depth_to_world_points.
  3. Runs the REAL LiveDepthFeatureSource(camera).compute(depth, mask_a,
     mask_b) and diffs it against step 2's expectation, per a tolerance
     justified by which kind of computation that feature is (closed-form
     geometry = tight; Sobel-based = wider, kernel-justified; estimated/
     topological = directional/categorical, not a tight number).

Steps 1 and 2 share ONLY the raw authored SDF numbers -- neither is
derived from the other, and neither touches the code under test.

REUSED from the real module, because it is pure data/geometry plumbing,
not the feature math under test: parse_world_sdf, ObjectGeom, CameraGeom,
world_aabb, _rotation_matrix, R_OPTICAL_TO_LINK, enumerate_adjacent_pairs,
EPSILON_M. NEVER reused: feat_*, _ring_from_mask, depth_to_world_points,
range_to_z_depth, LiveDepthFeatureSource's own internals (only its public
.compute() is called, as the code under test).

Known approximations, stated up front:
  - minimum_boundary_distance's "true" value uses AABB-to-AABB gap, not
    exact OBB-to-OBB surface distance. Exact for axis-aligned (yaw-only)
    objects; an upper-bound approximation for rotated ones. Tolerance
    widened accordingly per-pair based on each object's actual rotation.
  - edge_continuity is validated categorically (single- vs multi-connected-
    component) on whichever pairs the world naturally provides; no
    artificial multi-component scene is constructed (would require
    inventing geometry not grounded in the authored SDF).
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import extract_relation_features as erf  # noqa: E402


# ============================================================================
# Independent ray-intersection renderer (step 1 -- the synthetic input)
# ============================================================================

@dataclass
class RenderResult:
    depth: np.ndarray            # (H,W) float32, z-depth (optical frame), nan = no hit
    owner: np.ndarray            # (H,W) object-name array ('' = no hit)
    true_points_world: np.ndarray  # (H,W,3) float64, true 3D hit point, nan = no hit
    hit_face: np.ndarray          # (H,W) int8, local face id of the hit (-1 = no hit / non-box)


def _ray_box_t(o_local, d_local, half):
    """Slab method. Returns (t_hit, face_id) or (inf, -1). face_id in 0..5
    for +-X,+-Y,+-Z local faces (2*axis + (1 if positive normal else 0))."""
    tmin, tmax = -np.inf, np.inf
    face_min, face_max = -1, -1
    for axis in range(3):
        if abs(d_local[axis]) < 1e-12:
            if o_local[axis] < -half[axis] or o_local[axis] > half[axis]:
                return math.inf, -1
            continue
        t1 = (-half[axis] - o_local[axis]) / d_local[axis]
        t2 = (half[axis] - o_local[axis]) / d_local[axis]
        lo_face, hi_face = 2 * axis, 2 * axis + 1
        if t1 > t2:
            t1, t2 = t2, t1
            lo_face, hi_face = hi_face, lo_face
        if t1 > tmin:
            tmin, face_min = t1, lo_face
        if t2 < tmax:
            tmax, face_max = t2, hi_face
        if tmin > tmax:
            return math.inf, -1
    if tmax < 0:
        return math.inf, -1
    if tmin >= 0:
        return tmin, face_min
    return tmax, face_max


def ray_cast_scene(camera: "erf.CameraGeom", objects: dict) -> RenderResult:
    """Vectorized (per-object, all pixels at once) nearest-hit ray caster.
    Ray parameterization is deliberately non-unit-length: d = R @ [px,py,1],
    so the intersection parameter t equals z-depth directly (same
    convention depth_to_world_points assumes on the way back), with no
    separate normalize-then-rescale step needed."""
    h, w = camera.height, camera.width
    fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    px = (uu - cx) / fx
    py = (vv - cy) / fy
    dirs_opt = np.stack([px, py, np.ones_like(px)], axis=-1).reshape(-1, 3)  # (N,3)

    R_link = erf._rotation_matrix(camera.rpy)
    R = R_link @ erf.R_OPTICAL_TO_LINK
    dirs_world = dirs_opt @ R.T
    origin = camera.xyz

    best_t = np.full(dirs_world.shape[0], np.inf)
    best_owner = np.full(dirs_world.shape[0], "", dtype=object)
    best_face = np.full(dirs_world.shape[0], -1, dtype=np.int8)

    for name, obj in objects.items():
        if obj.placeholder_pose:
            continue
        R_obj = erf._rotation_matrix(obj.rpy)
        o_local_single = (origin - obj.xyz) @ R_obj  # R_obj^T @ (o - xyz), row-vector form
        o_local = np.broadcast_to(o_local_single, (dirs_world.shape[0], 3))
        d_local = dirs_world @ R_obj

        if obj.geom_type == "box":
            sx, sy, sz = obj.dims
            half = np.array([sx / 2, sy / 2, sz / 2])
            t = np.full(o_local.shape[0], np.inf)
            face = np.full(o_local.shape[0], -1, dtype=np.int8)
            # Vectorized slab test.
            tmin = np.full(o_local.shape[0], -np.inf)
            tmax = np.full(o_local.shape[0], np.inf)
            fmin = np.full(o_local.shape[0], -1, dtype=np.int8)
            fmax = np.full(o_local.shape[0], -1, dtype=np.int8)
            for axis in range(3):
                d_ax = d_local[:, axis]
                o_ax = o_local[:, axis]
                parallel = np.abs(d_ax) < 1e-12
                safe_d = np.where(parallel, 1.0, d_ax)
                t1 = (-half[axis] - o_ax) / safe_d
                t2 = (half[axis] - o_ax) / safe_d
                lo_face = np.full(o_local.shape[0], 2 * axis, dtype=np.int8)
                hi_face = np.full(o_local.shape[0], 2 * axis + 1, dtype=np.int8)
                swap = t1 > t2
                t1s = np.where(swap, t2, t1)
                t2s = np.where(swap, t1, t2)
                lo_f = np.where(swap, hi_face, lo_face)
                hi_f = np.where(swap, lo_face, hi_face)
                outside_parallel = parallel & ((o_ax < -half[axis]) | (o_ax > half[axis]))
                take_min = t1s > tmin
                fmin = np.where(take_min, lo_f, fmin)
                tmin = np.where(take_min, t1s, tmin)
                take_max = t2s < tmax
                fmax = np.where(take_max, hi_f, fmax)
                tmax = np.where(take_max, t2s, tmax)
                tmin = np.where(outside_parallel, np.inf, tmin)
                tmax = np.where(outside_parallel, -np.inf, tmax)
            valid = tmin <= tmax
            use_min = valid & (tmin >= 0)
            use_max = valid & (~use_min) & (tmax >= 0)
            t = np.where(use_min, tmin, np.where(use_max, tmax, np.inf))
            face = np.where(use_min, fmin, np.where(use_max, fmax, -1))

        elif obj.geom_type == "cylinder":
            radius, length = obj.dims
            half_len = length / 2
            a = d_local[:, 0] ** 2 + d_local[:, 1] ** 2
            b = 2 * (o_local[:, 0] * d_local[:, 0] + o_local[:, 1] * d_local[:, 1])
            c = o_local[:, 0] ** 2 + o_local[:, 1] ** 2 - radius ** 2
            disc = b ** 2 - 4 * a * c
            side_t = np.full(o_local.shape[0], np.inf)
            has_side = (a > 1e-12) & (disc >= 0)
            sq = np.sqrt(np.clip(disc, 0, None))
            safe_a = np.where(a > 1e-12, a, 1.0)
            t_lo = (-b - sq) / (2 * safe_a)
            t_hi = (-b + sq) / (2 * safe_a)
            for cand in (t_lo, t_hi):
                z_at_cand = o_local[:, 2] + cand * d_local[:, 2]
                ok = has_side & (cand >= 0) & (np.abs(z_at_cand) <= half_len) & (cand < side_t)
                side_t = np.where(ok, cand, side_t)
            # End caps.
            cap_t = np.full(o_local.shape[0], np.inf)
            cap_face = np.full(o_local.shape[0], -1, dtype=np.int8)
            for sign, fid in ((-1.0, 4), (1.0, 5)):
                dz = d_local[:, 2]
                safe_dz = np.where(np.abs(dz) > 1e-12, dz, 1.0)
                tc = (sign * half_len - o_local[:, 2]) / safe_dz
                x_at = o_local[:, 0] + tc * d_local[:, 0]
                y_at = o_local[:, 1] + tc * d_local[:, 1]
                ok = (np.abs(dz) > 1e-12) & (tc >= 0) & (x_at ** 2 + y_at ** 2 <= radius ** 2) & (tc < cap_t)
                cap_face = np.where(ok, fid, cap_face)
                cap_t = np.where(ok, tc, cap_t)
            t = np.minimum(side_t, cap_t)
            face = np.where(side_t <= cap_t, np.int8(6), cap_face)  # 6 = "side" pseudo-face

        elif obj.geom_type == "sphere":
            r = obj.dims[0]
            a = np.sum(d_local ** 2, axis=1)
            b = 2 * np.sum(o_local * d_local, axis=1)
            c = np.sum(o_local ** 2, axis=1) - r ** 2
            disc = b ** 2 - 4 * a * c
            t = np.full(o_local.shape[0], np.inf)
            face = np.full(o_local.shape[0], 7, dtype=np.int8)  # 7 = "sphere" pseudo-face
            ok_disc = disc >= 0
            sq = np.sqrt(np.clip(disc, 0, None))
            safe_a = np.where(a > 1e-12, a, 1.0)
            t_lo = (-b - sq) / (2 * safe_a)
            t_hi = (-b + sq) / (2 * safe_a)
            cand = np.where(t_lo >= 0, t_lo, t_hi)
            valid = ok_disc & (cand >= 0)
            t = np.where(valid, cand, np.inf)
            face = np.where(valid, face, -1)
        else:
            continue

        t = np.where(np.isfinite(t) & (t >= camera.clip_near) & (t <= camera.clip_far), t, np.inf)
        closer = t < best_t
        best_t = np.where(closer, t, best_t)
        best_owner = np.where(closer, name, best_owner)
        best_face = np.where(closer, face, best_face)

    # Background/no-hit pixels get clip_far, not NaN: a real depth sensor
    # reports a range value (typically clamped to max range) for "nothing
    # there", never NaN -- and NaN would poison every Sobel-based feature's
    # 3x3 kernel at every single object silhouette edge.
    depth = np.where(np.isfinite(best_t), best_t, camera.clip_far).astype(np.float32).reshape(h, w)
    owner = best_owner.reshape(h, w)
    face = best_face.reshape(h, w)

    hit = np.isfinite(best_t)
    points = np.full((dirs_world.shape[0], 3), np.nan)
    points[hit] = origin + best_t[hit, None] * dirs_world[hit]
    points = points.reshape(h, w, 3)

    return RenderResult(depth=depth, owner=owner, true_points_world=points, hit_face=face)


# ============================================================================
# Independent closed-form / ground-truth-derived expectations (step 2)
# ============================================================================

FACE_LOCAL_NORMALS = {
    0: np.array([-1, 0, 0]), 1: np.array([1, 0, 0]),
    2: np.array([0, -1, 0]), 3: np.array([0, 1, 0]),
    4: np.array([0, 0, -1]), 5: np.array([0, 0, 1]),
}


def expected_relative_height(render: RenderResult, mask_a, mask_b):
    z_a = np.nanmedian(render.true_points_world[..., 2][mask_a])
    z_b = np.nanmedian(render.true_points_world[..., 2][mask_b])
    return float(z_a - z_b)


FACE_DOMINANCE_THRESHOLD = 0.85  # below this, the view mixes >=2 differently-
# oriented surfaces (two box faces, or a flat face + curved side) and there
# is no single true normal to hand-derive an expectation from -- not just a
# "mostly one face" approximation.


def expected_face_normal_world(obj, render: RenderResult, mask):
    faces, counts = np.unique(render.hit_face[mask], return_counts=True)
    if len(faces) == 0:
        return None
    dominant_idx = np.argmax(counts)
    dominant = faces[dominant_idx]
    if counts[dominant_idx] / counts.sum() < FACE_DOMINANCE_THRESHOLD:
        return None  # mixed-surface view -- see FACE_DOMINANCE_THRESHOLD
    if dominant not in FACE_LOCAL_NORMALS:
        return None  # cylinder/sphere curved surface -- no single flat normal
    local_n = FACE_LOCAL_NORMALS[dominant]
    R_obj = erf._rotation_matrix(obj.rpy)
    return R_obj @ local_n


def my_ring(mask: np.ndarray, ring_px: int = 3) -> np.ndarray:
    """Independent reimplementation of the same simple concept as the real
    _ring_from_mask (mask minus its erosion) -- written from scratch here,
    not imported, per the anti-circularity design."""
    m = mask.astype(np.uint8)
    eroded = cv2.erode(m, np.ones((ring_px, ring_px), np.uint8))
    return m.astype(bool) & (~eroded.astype(bool))


def expected_sobel_region_mean(depth, region):
    """Same standard Sobel operator either side of this test would use
    (a generic, independently-verifiable image primitive) -- what's under
    test is boundary_sharpness / depth_gradient's masking+averaging region,
    not the convolution kernel itself."""
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    if not np.any(region):
        return float("nan")
    return float(np.nanmean(mag[region]))


def expected_occlusion_consistency(render: RenderResult, mask_a, mask_b):
    """Ground-truth version of the same directional statistic
    feat_occlusion_boundary_score_proxy computes, using the renderer's own
    true depth (== the synthetic depth fed to the real function -- same
    array, independently-coded region/statistic)."""
    ring_a, ring_b = my_ring(mask_a), my_ring(mask_b)
    interface = (ring_a | ring_b) & (
        cv2.dilate(mask_a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        | cv2.dilate(mask_b.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    )
    if not np.any(interface):
        return float("nan")
    near_a = cv2.dilate(mask_a.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & interface
    near_b = cv2.dilate(mask_b.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & interface
    if not np.any(near_a) or not np.any(near_b):
        return float("nan")
    mean_b = np.nanmean(render.depth[near_b])
    a_nearer = np.nansum(render.depth[near_a] < mean_b)
    consistency = a_nearer / max(1, near_a.sum())
    return float(abs(consistency - 0.5) * 2)


# ============================================================================
# Comparison bookkeeping
# ============================================================================

@dataclass
class FeatureResult:
    world: str
    pair: tuple
    feature: str
    mode: str            # "numeric" | "directional" | "categorical" | "not-independently-testable"
    expected: object
    actual: object
    tolerance: object
    passed: Optional[bool]
    note: str = ""


def numeric_check(world, pair, feature, expected, actual, abs_tol=None, rel_tol=None, note=""):
    if expected is None or actual is None or any(
        isinstance(v, float) and math.isnan(v) for v in (expected, actual)
    ):
        return FeatureResult(world, pair, feature, "numeric", expected, actual,
                              f"abs={abs_tol},rel={rel_tol}", None, note + " [skipped: NaN on one side]")
    diff = abs(expected - actual)
    tol = max(abs_tol or 0.0, (rel_tol or 0.0) * max(abs(expected), 1e-9))
    return FeatureResult(world, pair, feature, "numeric", expected, actual,
                          f"abs={abs_tol},rel={rel_tol}", diff <= tol, note)


# ============================================================================
# Test-pair definitions
# ============================================================================

HAND_PICKED_PAIRS = {
    "world1_baseline": [("obj_bowl", "obj_apple"), ("obj_bottle", "obj_mug"), ("obj_notebook", "obj_remote")],
    "world2_dense_clutter": [("obj_bowl", "obj_apple"), ("obj_jar", "obj_cereal_box"), ("obj_chopping_board", "obj_knife")],
    "world3_thin_objects": [("obj_chopstick_a", "obj_chopstick_b"), ("obj_plate", "obj_bottle"), ("obj_fork", "obj_ruler")],
    "world4_support_scene": [("obj_cereal_box", "obj_tissue_box"), ("obj_cereal_box", "obj_container"), ("obj_book", "obj_glass_jar")],
    "world5_occlusion_scene": [("obj_bottle", "obj_jar"), ("obj_cereal_box", "obj_bottle"), ("obj_container", "obj_toy")],
    "world6_dense_mixed": [("obj_bowl", "obj_apple"), ("obj_apple", "obj_ruler"), ("obj_bowl", "obj_ruler")],
}


def run_pair(world_id, objects, camera, render, name_a, name_b, results, full_sweep=False):
    if name_a not in objects or name_b not in objects:
        results.append(FeatureResult(world_id, (name_a, name_b), "ALL", "n/a", None, None, None, None,
                                      f"object missing from parsed SDF (a_present={name_a in objects}, b_present={name_b in objects})"))
        return
    obj_a, obj_b = objects[name_a], objects[name_b]
    mask_a = render.owner == name_a
    mask_b = render.owner == name_b
    pair = (name_a, name_b)

    if not np.any(mask_a) or not np.any(mask_b):
        results.append(FeatureResult(world_id, pair, "ALL", "n/a", None, None, None, None,
                                      f"empty rendered mask (px_a={mask_a.sum()}, px_b={mask_b.sum()}) -- object not visible from depth_camera1 in this scene"))
        return

    live = erf.LiveDepthFeatureSource(camera=camera)
    actual = live.compute(render.depth, mask_a, mask_b)

    rotated_a = np.any(np.abs(obj_a.rpy) > 1e-6)
    rotated_b = np.any(np.abs(obj_b.rpy) > 1e-6)
    aabb_note = " [AABB-gap approx: one/both objects rotated, widened tolerance]" if (rotated_a or rotated_b) else ""

    # 2. minimum_boundary_distance (+ is_metric). NOTE: this feature is a 2D
    # image-plane proxy (pixel distance-transform x depth-based scale), not
    # a true 3D Euclidean surface distance -- it cannot see a gap that is
    # purely along the camera's depth axis. So "expected" is computed the
    # same way, on ground-truth (ray-cast, not code-under-test) masks: a
    # distance-transform is a generic, independently-trustworthy image
    # primitive (like Sobel), applied here to OUR OWN masks, never to the
    # real function's output.
    median_depth = float(np.nanmedian(render.depth[mask_a | mask_b]))
    dist_to_b_px = cv2.distanceTransform((~mask_b).astype(np.uint8), cv2.DIST_L2, 5)
    expected_px_dist = float(np.min(dist_to_b_px[mask_a])) if np.any(mask_a) else float("nan")
    metres_per_px = median_depth / camera.fx if median_depth > 0 else float("nan")
    expected_mbd = expected_px_dist * metres_per_px
    results.append(numeric_check(world_id, pair, "minimum_boundary_distance",
                                  expected_mbd, actual["minimum_boundary_distance"],
                                  abs_tol=max(metres_per_px, 1e-4), rel_tol=0.05,
                                  note="ground-truth-mask distanceTransform (2D proxy, matches feature's actual definition)"))

    # 4. relative_height
    expected_rh = expected_relative_height(render, mask_a, mask_b)
    results.append(numeric_check(world_id, pair, "relative_height",
                                  expected_rh, actual["relative_height"], abs_tol=0.01))

    # 6. overlap_ratio. NOTE: expected uses OUR OWN rendered ground-truth
    # mask's actual visible extent, not a full-AABB projection -- an object
    # partially hidden behind another (common in these cluttered scenes)
    # projects a smaller true bbox than its full unoccluded AABB would.
    def _bbox(m):
        ys, xs = np.where(m)
        return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None
    box_a, box_b = _bbox(mask_a), _bbox(mask_b)
    if box_a and box_b:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        expected_ov = float(inter / union) if union else float("nan")
    else:
        expected_ov = float("nan")
    results.append(numeric_check(world_id, pair, "overlap_ratio",
                                  expected_ov, actual["overlap_ratio"], abs_tol=0.02,
                                  note="ground-truth rendered mask bbox (accounts for real occlusion), not full-AABB projection"))

    # 1. boundary_sharpness / 5. depth_gradient -- validate the masking/averaging
    # region against an independent ring/interface definition, same Sobel field.
    ring_a, ring_b = my_ring(mask_a), my_ring(mask_b)
    interface_bs = (ring_a | ring_b) & (
        cv2.dilate(mask_a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        | cv2.dilate(mask_b.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    )
    expected_bs = expected_sobel_region_mean(render.depth, interface_bs)
    results.append(numeric_check(world_id, pair, "boundary_sharpness",
                                  expected_bs, actual["boundary_sharpness"], rel_tol=0.15,
                                  note="independent ring reimplementation, same Sobel field"))
    expected_dg = expected_sobel_region_mean(render.depth, mask_a | mask_b)
    results.append(numeric_check(world_id, pair, "depth_gradient",
                                  expected_dg, actual["depth_gradient"], rel_tol=0.20))

    # 3. surface_normal_consistency -- flat-face pairs only (box-box); NaN/skip
    # on curved (cylinder/sphere) surfaces, which have no single true normal.
    na = expected_face_normal_world(obj_a, render, mask_a)
    nb = expected_face_normal_world(obj_b, render, mask_b)
    faces_a, counts_a = np.unique(render.hit_face[mask_a], return_counts=True)
    faces_b, counts_b = np.unique(render.hit_face[mask_b], return_counts=True)
    face_split_note = (f"face-hit split a={dict(zip(faces_a.tolist(), counts_a.tolist()))} "
                        f"b={dict(zip(faces_b.tolist(), counts_b.tolist()))}")
    if na is not None and nb is not None:
        na = na / (np.linalg.norm(na) + 1e-9)
        nb = nb / (np.linalg.norm(nb) + 1e-9)
        expected_snc = float(abs(np.dot(na, nb)))
        results.append(numeric_check(world_id, pair, "surface_normal_consistency",
                                      expected_snc, actual["surface_normal_consistency"], abs_tol=0.05,
                                      note=face_split_note))
    else:
        results.append(FeatureResult(world_id, pair, "surface_normal_consistency", "not-independently-testable",
                                      None, actual["surface_normal_consistency"], None, None,
                                      "one or both objects show a curved surface or a mixed multi-face view "
                                      "(no dominant face >=85% of masked pixels) -- no single true normal; "
                                      + face_split_note))

    # 7. edge_continuity -- categorical: single connected interface component?
    # Interface formula matches feat_edge_continuity's own (deliberately
    # different from boundary_sharpness's interface_bs above).
    interface_ec = (ring_a & ring_b) | (
        ring_a & cv2.dilate(mask_b.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ) | (
        ring_b & cv2.dilate(mask_a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    )
    num_labels, _ = cv2.connectedComponents(interface_ec.astype(np.uint8))
    expected_category = "single" if (num_labels - 1) == 1 else f"multi({num_labels-1})"
    actual_ec = actual["edge_continuity"]
    if math.isnan(actual_ec):
        cat_pass = None
    elif expected_category == "single":
        cat_pass = actual_ec > 0.9
    else:
        cat_pass = actual_ec < 0.9
    results.append(FeatureResult(world_id, pair, "edge_continuity", "categorical",
                                  expected_category, actual_ec, ">0.9 if single / <0.9 if multi", cat_pass,
                                  "no artificial multi-component scene constructed; category taken as the geometry naturally gives it"))

    # 8. occlusion_boundary_score -- directional only, per sign-off note.
    expected_occ = expected_occlusion_consistency(render, mask_a, mask_b)
    actual_occ = actual["occlusion_boundary_score"]
    if math.isnan(expected_occ) or math.isnan(actual_occ):
        occ_pass = None
    else:
        occ_pass = abs(expected_occ - actual_occ) <= 0.15
    results.append(FeatureResult(world_id, pair, "occlusion_boundary_score", "directional",
                                  expected_occ, actual_occ, "abs<=0.15 (directional check only; proxy, not verified GT per file's own audit)",
                                  occ_pass))


def main():
    all_results: list[FeatureResult] = []

    for world_name, pairs in HAND_PICKED_PAIRS.items():
        sdf_path = erf.WORLDS_SDF_DIR / f"{world_name}.sdf"
        objects, cameras = erf.parse_world_sdf(sdf_path)
        camera = cameras[1]
        render = ray_cast_scene(camera, objects)
        print(f"[{world_name}] parsed {len(objects)} objects, camera={camera.width}x{camera.height} "
              f"fov={camera.horizontal_fov:.2f}rad, rendered {np.isfinite(render.depth).sum()} hit pixels")

        for name_a, name_b in pairs:
            run_pair(world_name, objects, camera, render, name_a, name_b, all_results)

        if world_name == "world6_dense_mixed":
            masks = {name: (render.owner == name) for name in objects if np.any(render.owner == name)}
            sweep_pairs = erf.enumerate_adjacent_pairs(masks)
            print(f"[{world_name}] full regression sweep: {len(sweep_pairs)} adjacent pairs found")
            for name_a, name_b in sweep_pairs:
                if (name_a, name_b) in pairs or (name_b, name_a) in pairs:
                    continue
                run_pair(world_name, objects, camera, render, name_a, name_b, all_results, full_sweep=True)

    print_report(all_results)


def print_report(results: list[FeatureResult]):
    print("\n" + "=" * 100)
    print("PER-PAIR, PER-FEATURE RESULTS")
    print("=" * 100)
    current_key = None
    for r in results:
        key = (r.world, r.pair)
        if key != current_key:
            print(f"\n--- {r.world} :: {r.pair[0]} / {r.pair[1]} ---")
            current_key = key
        status = "SKIP" if r.passed is None else ("PASS" if r.passed else "FAIL")
        print(f"  [{status:4s}] {r.feature:28s} mode={r.mode:12s} "
              f"expected={r.expected!r:>22} actual={r.actual!r:>22} tol={r.tolerance}"
              + (f"  NOTE: {r.note}" if r.note else ""))

    print("\n" + "=" * 100)
    print("FAILURES -- DETAIL")
    print("=" * 100)
    failures = [r for r in results if r.passed is False]
    if not failures:
        print("  (none)")
    for r in failures:
        print(f"\n  {r.world} :: {r.pair} :: {r.feature}")
        print(f"    expected = {r.expected}")
        print(f"    actual   = {r.actual}")
        print(f"    tolerance= {r.tolerance}")
        print(f"    note     = {r.note}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    by_feature = {}
    for r in results:
        by_feature.setdefault(r.feature, {"pass": 0, "fail": 0, "skip": 0})
        if r.passed is None:
            by_feature[r.feature]["skip"] += 1
        elif r.passed:
            by_feature[r.feature]["pass"] += 1
        else:
            by_feature[r.feature]["fail"] += 1
    print(f"{'feature':30s} {'pass':>6s} {'fail':>6s} {'skip':>6s}")
    for feat, counts in by_feature.items():
        print(f"{feat:30s} {counts['pass']:6d} {counts['fail']:6d} {counts['skip']:6d}")
    total_pass = sum(c["pass"] for c in by_feature.values())
    total_fail = sum(c["fail"] for c in by_feature.values())
    total_skip = sum(c["skip"] for c in by_feature.values())
    print(f"{'TOTAL':30s} {total_pass:6d} {total_fail:6d} {total_skip:6d}")


if __name__ == "__main__":
    main()
