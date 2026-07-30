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


def get_prg_edge_error(rgb, depth, target_name, all_object_masks, trial_id,
                        camera_intrinsics=None, masks_are_bbox_placeholder=False):
    """
    target_name: key into all_object_masks identifying the grasp target.
    all_object_masks: dict[str, np.ndarray] — every detected object's mask
                       in-frame (real segmentation, or bbox_to_mask placeholders).
    camera_intrinsics: her CameraIntrinsics instance, built from Orbbec calibration.

    Returns a LIST of per-neighbor-pair results (one entry per target-adjacent
    edge), per collaborator's recommendation — logs all pairs, doesn't
    pre-select "the" neighbor.
    """
    config = _load_config()

    if config["use_stub"] or camera_intrinsics is None or target_name not in all_object_masks:
        return [{
            "prg_edge_error": _seeded_stub_value(trial_id),
            "is_stub": True,
            "relation_prediction": None,
            "relation_confidence": None,
            "geometric_feature_vector": None,
            "neighbor_name": None,
        }]

    try:
        from relation_classifier.scripts.extract_relation_features import (
            LiveDepthFeatureSource, enumerate_adjacent_pairs,
        )
        from relation_classifier.scripts.classify_relation import classify_relation
        from relation_classifier.scripts.compute_prg_edge_error import compute_prg_edge_error

        pairs = enumerate_adjacent_pairs(all_object_masks)  # confirmed: dict[str, mask] in, all overlapping pairs out
        target_pairs = [p for p in pairs if target_name in p]

        if not target_pairs:
            return []  # target has no adjacent objects this frame — valid, not an error

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
                "prg_edge_error": edge_error,
                "is_stub": False,
                "relation_prediction": relation,
                "relation_confidence": confidence,
                "geometric_feature_vector": features,
                "neighbor_name": neighbor_name,
                "mask_is_bbox_placeholder": masks_are_bbox_placeholder,
            })
        return results

    except Exception as e:
        return [{
            "prg_edge_error": None,
            "is_stub": True,
            "relation_prediction": None,
            "relation_confidence": None,
            "geometric_feature_vector": None,
            "neighbor_name": None,
            "error_note": str(e),
        }]