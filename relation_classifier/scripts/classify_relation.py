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

MODEL VERSIONING (added for the retrain/relation-classifier work):
`classify_relation()` itself is UNCHANGED in signature and behavior --
same 2-tuple (relation, confidence) return, bit-for-bit identical output
on legacy inputs (see tests/test_relation_model_v2_regression.py). Which
model actually runs is controlled by `relation_classifier/relation_model_config.json`'s
`relation_model_version` key ("legacy" default, or "v2"), NEVER by
integration/prg_config.json (that file's `use_stub` key is unrelated and
must never be touched by this switch). `classify_relation_versioned()` is
a new, additive function for callers that want to know which model
version actually served a given call (e.g. for schema logging) --
existing callers of `classify_relation()` are unaffected.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np

_LOGGER = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "relation_model_config.json"


def _load_model_version() -> str:
    """Never raises -- an absent or malformed config file means 'legacy',
    the safe default, not a crash. This is the ONLY place the version
    switch is read from; integration/prg_config.json's use_stub is a
    completely separate, untouched concern."""
    try:
        cfg = json.loads(_CONFIG_PATH.read_text())
        version = cfg.get("relation_model_version", "legacy")
        if version not in ("legacy", "v2"):
            _LOGGER.warning(
                "relation_model_config.json has unrecognized relation_model_version=%r "
                "-- falling back to 'legacy'.", version)
            return "legacy"
        return version
    except FileNotFoundError:
        return "legacy"
    except (json.JSONDecodeError, OSError) as e:
        _LOGGER.warning("Could not read %s (%s) -- falling back to 'legacy'.", _CONFIG_PATH, e)
        return "legacy"


_RETRAIN_MODELS_DIR = Path(__file__).resolve().parents[2] / "retrain" / "models"

# Unchanged from before this session, verbatim -- these are CWD-relative
# (assumes invocation from the project root, exactly as before), kept
# byte-identical to the pre-v2 version rather than switched to
# __file__-relative resolution, so there is zero risk of any path-
# resolution behavior difference sneaking into the legacy regression test.
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

# v2 artifacts are loaded LAZILY (only when relation_model_version=="v2" is
# actually requested) so that "legacy" stays fully usable even before any
# retrain has ever produced v2 files -- default behavior must never depend
# on retrain/models/ existing.
_V2_CACHE: dict = {}


def _load_v2():
    if _V2_CACHE:
        return _V2_CACHE
    metadata = json.loads((_RETRAIN_MODELS_DIR / "relation_binary_v2_metadata.json").read_text())
    _V2_CACHE["metadata"] = metadata
    _V2_CACHE["model"] = joblib.load(_RETRAIN_MODELS_DIR / "relation_binary_v2.joblib")
    _V2_CACHE["scaler"] = joblib.load(_RETRAIN_MODELS_DIR / "relation_binary_v2_scaler.joblib")
    if metadata["architecture"] == "two_stage":
        _V2_CACHE["support_model"] = joblib.load(_RETRAIN_MODELS_DIR / "support_v2.joblib")
        _V2_CACHE["support_scaler"] = joblib.load(_RETRAIN_MODELS_DIR / "support_v2_scaler.joblib")
    return _V2_CACHE


def _check_ood(x: np.ndarray, scaler, feature_names: list[str], model_version: str) -> None:
    """Logs (does not raise, does not change any return value) when any
    feature is more than 4 SD outside the training range this model's
    scaler was fit on -- added per R4/R6: the model must not fail
    silently on out-of-distribution input."""
    z = (x - scaler.mean_) / scaler.scale_
    max_abs_z = float(np.max(np.abs(z)))
    if max_abs_z > 4.0:
        worst_idx = int(np.argmax(np.abs(z)))
        _LOGGER.warning(
            "classify_relation (model_version=%s): feature %r is %.1f SD outside "
            "the training distribution (z=%.1f) -- prediction may be unreliable. "
            "Full z-scores: %s",
            model_version, feature_names[worst_idx], max_abs_z, z[worst_idx],
            dict(zip(feature_names, np.round(z, 2).tolist())),
        )


def _predict_binary(feature_dict: dict) -> tuple[int, float]:
    x = np.array([[feature_dict[c] for c in FEATURE_COLUMNS]])
    if np.isnan(x).any():
        raise ValueError(f"NaN in features, cannot classify: {feature_dict}")
    _check_ood(x[0], _SCALER, FEATURE_COLUMNS, "legacy")
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


def _classify_relation_legacy(feature_dict: dict) -> tuple[str, float]:
    pred, confidence = _predict_binary(feature_dict)
    if pred == 0:
        return "Separate", confidence
    return _predict_support_vs_contact(feature_dict), confidence


def _classify_relation_v2(feature_dict: dict) -> tuple[str, float]:
    v2 = _load_v2()
    x = np.array([[feature_dict[c] for c in FEATURE_COLUMNS]])
    if np.isnan(x).any():
        raise ValueError(f"NaN in features, cannot classify: {feature_dict}")
    _check_ood(x[0], v2["scaler"], FEATURE_COLUMNS, "v2")
    x_scaled = v2["scaler"].transform(x)
    model = v2["model"]

    if v2["metadata"]["architecture"] == "direct_3class":
        pred = model.predict(x_scaled)[0]
        proba = float(np.max(model.predict_proba(x_scaled)[0]))
        return str(pred), proba

    # two_stage
    pred = int(model.predict(x_scaled)[0])
    confidence = float(model.predict_proba(x_scaled)[0][pred])
    if pred == 0:
        return "Separate", confidence
    values = [feature_dict[c] for c in _SUPPORT_FEATURE_COLUMNS]
    values[_SUPPORT_FEATURE_COLUMNS.index("relative_height")] = abs(
        values[_SUPPORT_FEATURE_COLUMNS.index("relative_height")]
    )
    xs_scaled = v2["support_scaler"].transform(np.array([values]))
    sub_pred = int(v2["support_model"].predict(xs_scaled)[0])
    return ("Support" if sub_pred == 1 else "Contact"), confidence


def classify_relation_versioned(feature_dict: dict) -> dict:
    """Additive: returns {"relation", "confidence", "model_version"}.
    classify_relation() below is a thin wrapper that discards
    model_version, so ITS return contract never changes regardless of
    which version actually served the call."""
    version = _load_model_version()
    if version == "v2":
        relation, confidence = _classify_relation_v2(feature_dict)
    else:
        relation, confidence = _classify_relation_legacy(feature_dict)
    return {"relation": relation, "confidence": confidence, "model_version": version}


def classify_relation(feature_dict: dict) -> tuple[str, float]:
    result = classify_relation_versioned(feature_dict)
    return result["relation"], result["confidence"]
