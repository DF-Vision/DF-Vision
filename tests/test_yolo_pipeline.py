"""
tests/test_yolo_pipeline.py

C3: YOLO pipeline full verification. See project_closure/CRITERIA.md's
C3 section for the pre-committed acceptance criteria. Covers (a) label
round trip, (b) class index single source of truth, (c) grouped split,
(e) per-object gate metric computation + config-driven thresholds. (d)
the real 2-epoch training run is a separate, slower test
(test_yolo_real_training.py) so this file stays fast.
"""
import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = PROJECT_ROOT / "perception-calibration-gap" / "yolo"
sys.path.insert(0, str(YOLO_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "real_camera"))

import auto_label_from_markers as alm
from class_map import load_class_names, load_class_index
import train_yolo_scaffold as tys
import detection_metrics as dm
from per_object_gate import evaluate_gate, _load_thresholds

MARKER_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)


def _synthetic_marker_image(marker_id: int, tx0: int, ty0: int, marker_px: int = 120,
                             canvas_size=(480, 640)):
    marker_img = cv2.aruco.generateImageMarker(MARKER_DICT, marker_id, marker_px)
    canvas = np.full((*canvas_size, 3), 200, dtype=np.uint8)
    canvas[ty0:ty0 + marker_px, tx0:tx0 + marker_px] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    return canvas


# --- (a) label round trip ---

@pytest.mark.parametrize("object_name,marker_id,tx0,ty0", [
    ("apple", 0, 200, 150),
    ("purse", 2, 80, 60),
    ("glass", 3, 350, 220),
])
def test_auto_label_round_trip_matches_known_box_within_1px(object_name, marker_id, tx0, ty0):
    canvas = _synthetic_marker_image(marker_id, tx0, ty0)
    lines, raw = alm.auto_label_image(canvas, dictionary=MARKER_DICT, dilate_px=60)
    assert len(lines) == 1
    assert len(raw) == 1
    class_id, obj_name, bbox_px = raw[0]
    assert obj_name == object_name
    assert class_id == load_class_index()[object_name]

    # Decode the WRITTEN label line back to pixels (normalize/denormalize round trip)
    decoded_class_id, decoded_bbox = alm.yolo_line_to_bbox_px(lines[0], canvas.shape)
    assert decoded_class_id == class_id
    for a, b in zip(bbox_px, decoded_bbox):
        assert abs(a - b) < 1.0, f"round-trip box {decoded_bbox} vs known {bbox_px} exceeds 1px"


def test_auto_label_skips_non_study_object_markers():
    """occluder_A (marker_id=5) is not in class_map.json -- must be
    skipped, not mislabeled as some study object."""
    canvas = _synthetic_marker_image(5, 200, 150)  # occluder_A
    lines, raw = alm.auto_label_image(canvas, dictionary=MARKER_DICT, dilate_px=60)
    assert lines == []
    assert raw == []


def test_contact_sheet_written(tmp_path):
    """SYNTHETIC contact sheet -- boxes drawn onto images, saved for
    visual inspection."""
    images_and_raw = []
    for object_name, marker_id, tx0, ty0 in [("apple", 0, 100, 100), ("purse", 2, 300, 100)]:
        canvas = _synthetic_marker_image(marker_id, tx0, ty0)
        _, raw = alm.auto_label_image(canvas, dictionary=MARKER_DICT, dilate_px=60)
        images_and_raw.append((canvas, raw))
    out_path = tmp_path / "contact_sheet_SYNTHETIC.png"
    alm.draw_contact_sheet(images_and_raw, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


# --- (b) class index single source of truth ---

def test_class_map_is_single_source_used_by_trainer_and_labeler():
    names = load_class_names()
    assert tys.STUDY_OBJECT_NAMES == names, "trainer's STUDY_OBJECT_NAMES must come from class_map.json, not its own list"
    index = load_class_index()
    assert index == {n: i for i, n in enumerate(names)}


def test_data_yaml_names_list_matches_class_map(tmp_path):
    data_yaml_path = tys.generate_synthetic_yolo_dataset(str(tmp_path / "ds"), n_images=8, seed=0)
    text = Path(data_yaml_path).read_text()
    names_line = [l for l in text.splitlines() if l.startswith("names:")][0]
    written_names = eval(names_line.split("names:", 1)[1].strip())
    assert written_names == load_class_names()


# --- (c) grouped split by session ---

def test_grouped_split_has_zero_session_overlap(tmp_path):
    data_yaml, session_of_image, train_sessions, val_sessions = tys.generate_synthetic_yolo_dataset_grouped(
        str(tmp_path / "grouped_ds"), n_sessions=12, images_per_session=3, seed=7,
    )
    assert train_sessions.isdisjoint(val_sessions), "a session must never appear in both splits"
    assert len(train_sessions) + len(val_sessions) == 12

    # cross-check against the files actually written on disk
    root = Path(data_yaml).parent
    train_stems = {p.stem for p in (root / "images" / "train").glob("*.jpg")}
    val_stems = {p.stem for p in (root / "images" / "val").glob("*.jpg")}
    train_stem_sessions = {session_of_image[s] for s in train_stems}
    val_stem_sessions = {session_of_image[s] for s in val_stems}
    assert train_stem_sessions == train_sessions
    assert val_stem_sessions == val_sessions
    assert train_stem_sessions.isdisjoint(val_stem_sessions)


def test_grouped_split_is_deterministic_under_seed(tmp_path):
    _, _, train_a, val_a = tys.generate_synthetic_yolo_dataset_grouped(
        str(tmp_path / "ds_a"), n_sessions=10, images_per_session=2, seed=42,
    )
    _, _, train_b, val_b = tys.generate_synthetic_yolo_dataset_grouped(
        str(tmp_path / "ds_b"), n_sessions=10, images_per_session=2, seed=42,
    )
    assert train_a == train_b
    assert val_a == val_b


def test_grouped_split_different_seed_can_differ(tmp_path):
    _, _, train_a, _ = tys.generate_synthetic_yolo_dataset_grouped(
        str(tmp_path / "ds_c"), n_sessions=10, images_per_session=2, seed=1,
    )
    _, _, train_b, _ = tys.generate_synthetic_yolo_dataset_grouped(
        str(tmp_path / "ds_d"), n_sessions=10, images_per_session=2, seed=2,
    )
    assert train_a != train_b, "different seeds should (with high probability) produce different splits"


# --- (e) per-object gate: known-answer detection-metrics tests ---

def test_detection_metrics_known_answer_perfect_match():
    """Hand-computed: 2 predictions exactly matching 2 ground-truth boxes
    (IoU=1.0 each) -> TP=2, FP=0, FN=0, precision=1.0, recall=1.0, AP50=1.0."""
    gts = [(0, 0, 10, 10), (20, 20, 30, 30)]
    preds = [(0, 0, 10, 10), (20, 20, 30, 30)]
    m = dm.compute_detection_metrics(preds, gts, iou_thresh=0.5)
    assert m.tp == 2 and m.fp == 0 and m.fn == 0
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.ap50 == pytest.approx(1.0)


def test_detection_metrics_known_answer_one_miss_one_false_positive():
    """Hand-computed: 1 GT box matched (IoU=1.0), 1 GT box missed entirely
    (no prediction), 1 spurious prediction with no matching GT (IoU=0 vs
    both GTs). By hand: TP=1, FP=1, FN=1 -> precision=1/2=0.5, recall=1/2=0.5.
    AP50 (rank order: first pred is the perfect match, second is the FP):
    rank1: TP cumulative -> precision=1.0, recall=0.5
    rank2: FP cumulative -> precision=0.5, recall=0.5
    precision envelope (non-increasing from the right): [1.0, 0.5] (already
    non-increasing, unchanged). AP = area under (recall, envelope), recall
    points [0, 0.5, 0.5] with envelope [1.0, 1.0, 0.5] pointwise trapezoid:
    segment 0->0.5 at envelope=1.0 contributes 0.5*1.0=0.5; segment 0.5->0.5
    contributes 0 (zero width). AP50 = 0.5."""
    gts = [(0, 0, 10, 10), (50, 50, 60, 60)]  # second GT never gets a matching prediction
    preds = [(0, 0, 10, 10), (100, 100, 110, 110)]  # first matches, second is a spurious FP
    m = dm.compute_detection_metrics(preds, gts, iou_thresh=0.5)
    assert m.tp == 1 and m.fp == 1 and m.fn == 1
    assert m.precision == pytest.approx(0.5)
    assert m.recall == pytest.approx(0.5)
    assert m.ap50 == pytest.approx(0.5, abs=1e-6)


def test_compute_per_object_metrics_feeds_gate_correctly():
    gts_by_object = {
        "apple": [(0, 0, 10, 10), (20, 20, 30, 30)],
        "purse": [(0, 0, 10, 10)],
    }
    preds_by_object = {
        "apple": [(0, 0, 10, 10), (20, 20, 30, 30)],  # perfect
        "purse": [(100, 100, 110, 110)],  # complete miss
    }
    metrics = dm.compute_per_object_metrics(preds_by_object, gts_by_object)
    report = evaluate_gate(metrics)
    assert "apple" not in report.failed_objects
    assert "purse" in report.failed_objects


# --- (e) thresholds moved to config, legacy behavior unchanged ---

def test_thresholds_loaded_from_config_match_legacy_defaults():
    thresholds = _load_thresholds()
    assert thresholds == {"precision_threshold": 0.5, "recall_threshold": 0.5, "ap50_threshold": 0.5}


def test_missing_config_falls_back_to_same_defaults(tmp_path):
    thresholds = _load_thresholds(config_path=tmp_path / "does_not_exist.json")
    assert thresholds == {"precision_threshold": 0.5, "recall_threshold": 0.5, "ap50_threshold": 0.5}
