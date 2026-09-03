import json, random, hashlib, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = REPO_ROOT / "integration" / "prg_config.json"


def _load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _stable_hash(trial_id):
    return int(hashlib.sha256(trial_id.encode()).hexdigest(), 16) % (2**32)


def _seeded_stub_value(trial_id):
    rng = random.Random(_stable_hash(trial_id))
    return rng.uniform(0.0, 1.0)


def bbox_to_mask(bbox, shape):
    """Placeholder only — flags mask_is_bbox_placeholder=True downstream.
    boundary_sharpness / edge_continuity / occlusion_boundary_score will be
    distorted on non-rectangular objects (confirmed by collaborator)."""
    import numpy as np
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    mask[y1:y2, x1:x2] = True
    return mask


def _resolve_prg_edges(rgb, depth, target_name, all_object_masks, trial_id,
                        camera_intrinsics=None, masks_are_bbox_placeholder=False):
    """
    Internal. Computes the per-edge results AND the trial-level stub/error
    status in one place, so the two can never drift out of sync with each
    other. Every returned edge dict has EXACTLY SCHEMA.md Section 3b's six
    fields (neighbor_name, relation_prediction, relation_confidence,
    prg_edge_error, geometric_feature_vector, mask_is_bbox_placeholder) --
    no extra keys, on every code path (stub, real, exception-fallback).

    Returns (edges: list[dict], is_stub: bool, error_note: Optional[str]).
    """
    config = _load_config()

    if config["use_stub"] or camera_intrinsics is None or target_name not in all_object_masks:
        edge = {
            "neighbor_name": None,
            "relation_prediction": None,
            "relation_confidence": None,
            "prg_edge_error": _seeded_stub_value(trial_id),
            "geometric_feature_vector": None,
            "mask_is_bbox_placeholder": masks_are_bbox_placeholder,
        }
        return [edge], True, None

    try:
        from relation_classifier.scripts.extract_relation_features import (
            LiveDepthFeatureSource, enumerate_adjacent_pairs,
        )
        from relation_classifier.scripts.classify_relation import classify_relation
        from relation_classifier.scripts.compute_prg_edge_error import compute_prg_edge_error

        pairs = enumerate_adjacent_pairs(all_object_masks)  # confirmed: dict[str, mask] in, all overlapping pairs out
        target_pairs = [p for p in pairs if target_name in p]

        if not target_pairs:
            return [], False, None  # target has no adjacent objects this frame — valid, not an error

        feature_source = LiveDepthFeatureSource(camera_intrinsics)
        results = []
        for (name_a, name_b) in target_pairs:
            neighbor_name = name_b if name_a == target_name else name_a
            mask_a = all_object_masks[target_name]
            mask_b = all_object_masks[neighbor_name]

            features = feature_source.compute(depth, mask_a, mask_b)
            relation, confidence = classify_relation(features)
            edge_error = compute_prg_edge_error(
                rgb, depth, target_bbox=None, relation=relation, confidence=confidence,
                mask_a=mask_a, mask_b=mask_b,
            )
            assert 0.0 <= edge_error <= 1.0, f"prg_edge_error out of range: {edge_error}"

            results.append({
                "neighbor_name": neighbor_name,
                "relation_prediction": relation,
                "relation_confidence": confidence,
                "prg_edge_error": edge_error,
                "geometric_feature_vector": features,
                "mask_is_bbox_placeholder": masks_are_bbox_placeholder,
            })
        return results, False, None

    except Exception as e:
        edge = {
            "neighbor_name": None,
            "relation_prediction": None,
            "relation_confidence": None,
            # Documented neutral fallback (SCHEMA.md Section 3b: "Always a
            # scalar, never None") -- matches what the real
            # compute_prg_edge_error() does on ambiguous/insufficient input,
            # so a crash here can't propagate a None into
            # conformal_gate.py's nonconformity_dual(), which divides by a
            # percentile threshold.
            "prg_edge_error": 0.5,
            "geometric_feature_vector": None,
            "mask_is_bbox_placeholder": masks_are_bbox_placeholder,
        }
        return [edge], True, str(e)


def get_prg_edge_error(rgb, depth, target_name, all_object_masks, trial_id,
                        camera_intrinsics=None, masks_are_bbox_placeholder=False):
    """
    target_name: key into all_object_masks identifying the grasp target.
    all_object_masks: dict[str, np.ndarray] — every detected object's mask
                       in-frame (real segmentation, or bbox_to_mask placeholders).
    camera_intrinsics: her CameraIntrinsics instance, built from Orbbec calibration.

    Returns a LIST of per-neighbor-pair results (one entry per target-adjacent
    edge), per collaborator's recommendation — logs all pairs, doesn't
    pre-select "the" neighbor. Every entry matches SCHEMA.md Section 3b
    exactly (see _resolve_prg_edges).
    """
    edges, _is_stub, _error_note = _resolve_prg_edges(
        rgb, depth, target_name, all_object_masks, trial_id,
        camera_intrinsics, masks_are_bbox_placeholder,
    )
    return edges


def log_integration_fields(rgb, depth, target_name, all_object_masks, trial_id,
                            camera_intrinsics=None, masks_are_bbox_placeholder=False) -> dict:
    """
    Single entry point for the three TrialRecord fields this integration
    owns (SCHEMA.md Section 3a): prg_edges, prg_is_stub, prg_error_note.
    Computes them together (via _resolve_prg_edges) so callers (run_trial.py)
    never derive prg_is_stub/prg_error_note by hand from prg_edges' contents
    -- which is no longer even possible now that per-edge dicts carry only
    SCHEMA.md's six documented fields, nothing else.
    """
    edges, is_stub, error_note = _resolve_prg_edges(
        rgb, depth, target_name, all_object_masks, trial_id,
        camera_intrinsics, masks_are_bbox_placeholder,
    )
    return {"prg_edges": edges, "prg_is_stub": is_stub, "prg_error_note": error_note}