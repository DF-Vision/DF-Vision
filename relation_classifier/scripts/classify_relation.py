#!/usr/bin/env python3
"""
classify_relation.py
---------------------
Wraps the frozen relation classifier (logistic_model.joblib + scaler.joblib).
Outputs three-way Contact/Support/Separate: Contact/Separate come from the
model's actual trained target (gazebo_contact, physics-verified touching).
Support is a second-stage decision, made only for pairs the first model
calls touching, by a separate 3-feature Logistic Regression sub-classifier
(support_logistic_model.joblib + support_scaler.joblib) trained on
heuristic_label-derived Contact/Support ground truth from
training_data_metric_only.csv (176 rows, group-aware 5-fold CV:
F1(Support)=0.659, balanced accuracy=0.706 -- see
relation_classifier/models/support_summary.txt for the full sweep,
baseline comparisons, and McNemar's-test result this replaced a single
SUPPORT_RELATIVE_HEIGHT_THRESHOLD cutoff on relative_height with: that
threshold capped out at balanced accuracy 0.529 (barely above chance)
despite a misleadingly reasonable-looking F1 score, because a single
threshold on relative_height's magnitude cannot separate the two classes
in this data -- their distributions substantially overlap.
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

# Support-vs-Contact sub-classifier. relative_height is fed in as its
# absolute value: object_a/object_b ordering upstream is set by SDF
# document order, not physical stacking order, so raw signed
# relative_height does not consistently mean "target below neighbor"
# (confirmed during tuning: 8/75 labeled Support pairs have positive raw
# relative_height) -- this must match how the model was trained.
_SUPPORT_FEATURE_COLUMNS = ["minimum_boundary_distance", "overlap_ratio", "relative_height"]
_SUPPORT_MODEL = joblib.load("relation_classifier/models/support_logistic_model.joblib")
_SUPPORT_SCALER = joblib.load("relation_classifier/models/support_scaler.joblib")


def _predict_binary(feature_dict: dict) -> tuple[int, float]:
    x = np.array([[feature_dict[c] for c in FEATURE_COLUMNS]])
    if np.isnan(x).any():
        raise ValueError(f"NaN in features, cannot classify: {feature_dict}")
    x_scaled = _SCALER.transform(x)
    pred = int(_MODEL.predict(x_scaled)[0])
    proba = _MODEL.predict_proba(x_scaled)[0][pred]
    return pred, float(proba)


def _predict_support_vs_contact(feature_dict: dict) -> str:
    values = [feature_dict[c] for c in _SUPPORT_FEATURE_COLUMNS]
    values[_SUPPORT_FEATURE_COLUMNS.index("relative_height")] = abs(
        values[_SUPPORT_FEATURE_COLUMNS.index("relative_height")]
    )
    x_scaled = _SUPPORT_SCALER.transform(np.array([values]))
    pred = int(_SUPPORT_MODEL.predict(x_scaled)[0])
    return "Support" if pred == 1 else "Contact"


def classify_relation(feature_dict: dict) -> tuple[str, float]:
    pred, confidence = _predict_binary(feature_dict)
    if pred == 0:
        return "Separate", confidence
    return _predict_support_vs_contact(feature_dict), confidence