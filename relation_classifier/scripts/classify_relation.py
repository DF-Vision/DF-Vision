#!/usr/bin/env python3
"""
classify_relation.py
---------------------
Wraps the frozen relation classifier (logistic_model.joblib + scaler.joblib).
Outputs three-way Contact/Support/Separate: Contact/Separate come from the
model's actual trained target (gazebo_contact, physics-verified touching);
Support is a post-hoc split of the Contact class using relative_height,
since the model was never trained to distinguish Support from Contact.
See training_data_v3_metric_only.csv heuristic_label stats: Support mean
relative_height=-0.0116, Contact=+0.0095, Separate=+0.0248.
"""

import joblib
import numpy as np

_MODEL = joblib.load("relation_classifier/models/logistic_model.joblib")
_SCALER = joblib.load("relation_classifier/models/scaler.joblib")

FEATURE_COLUMNS = [
    "boundary_sharpness", "minimum_boundary_distance", "surface_normal_consistency",
    "relative_height", "depth_gradient", "overlap_ratio", "edge_continuity",
    "occlusion_boundary_score",
]

# Placeholder — tune in Step 6 validation before trusting on hardware.
SUPPORT_RELATIVE_HEIGHT_THRESHOLD = -0.01


def _predict_binary(feature_dict: dict) -> tuple[int, float]:
    x = np.array([[feature_dict[c] for c in FEATURE_COLUMNS]])
    if np.isnan(x).any():
        raise ValueError(f"NaN in features, cannot classify: {feature_dict}")
    x_scaled = _SCALER.transform(x)
    pred = int(_MODEL.predict(x_scaled)[0])
    proba = _MODEL.predict_proba(x_scaled)[0][pred]
    return pred, float(proba)


def classify_relation(feature_dict: dict) -> tuple[str, float]:
    pred, confidence = _predict_binary(feature_dict)
    if pred == 0:
        return "Separate", confidence
    rel_h = feature_dict["relative_height"]
    if rel_h < SUPPORT_RELATIVE_HEIGHT_THRESHOLD:
        return "Support", confidence
    return "Contact", confidence