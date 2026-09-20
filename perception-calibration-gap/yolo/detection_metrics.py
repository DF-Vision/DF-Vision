"""
detection_metrics.py

C3 (project/closure session): computes precision/recall/AP50 FROM raw
predicted vs. ground-truth boxes (IoU matching), rather than
per_object_gate.py's existing tests which only ever fed it
already-computed metric dicts. This is the layer that would sit between
a real ultralytics val() run (or auto_label_from_markers.py's output
compared against known ground truth) and per_object_gate.py's gate logic.

Precision/recall here are computed at a single "all detections count"
operating point (this project has no confidence-gradation in its
synthetic fixtures to sweep a PR curve over multiple thresholds) --
TP/(TP+FP) and TP/(TP+FN) after greedy IoU-threshold matching. AP50 uses
the standard PASCAL VOC-style precision-envelope area under the PR curve
built by sweeping detections in confidence order (ties broken by input
order) -- with only one operating point per class in the tiny fixtures
this module is tested against, AP50 reduces to precision*recall's
interpolated-envelope area, verified by hand in
tests/test_detection_metrics.py's docstrings.
"""
from __future__ import annotations

from dataclasses import dataclass


def iou(box_a, box_b) -> float:
    """box = (x0, y0, x1, y1)."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class DetectionMatchResult:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    ap50: float


def _precision_recall_at_each_rank(preds_sorted, gts, iou_thresh: float):
    """Greedy matching in the given (already confidence-sorted) order;
    each GT box matched at most once. Returns per-rank cumulative
    (precision, recall) pairs, TP/FP/FN totals."""
    matched_gt = [False] * len(gts)
    precisions, recalls = [], []
    tp_cum = 0
    fp_cum = 0
    n_gt = len(gts)

    for pred_box in preds_sorted:
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gts):
            if matched_gt[j]:
                continue
            i = iou(pred_box, gt_box)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_iou >= iou_thresh and best_j >= 0:
            matched_gt[best_j] = True
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_gt if n_gt > 0 else 0.0)

    tp = tp_cum
    fp = fp_cum
    fn = n_gt - tp
    return precisions, recalls, tp, fp, fn


def _ap_from_pr(precisions, recalls) -> float:
    """PASCAL VOC-style AP: precision envelope (monotonically
    non-increasing from the right) integrated over recall via the
    trapezoid rule across the observed recall points (00, r1, r2, ..., 1
    implicitly at whatever max recall was reached)."""
    if not precisions:
        return 0.0
    # Precision envelope: for each recall level, take the max precision
    # at that recall or higher (standard PASCAL VOC continuous AP).
    envelope = list(precisions)
    for i in range(len(envelope) - 2, -1, -1):
        envelope[i] = max(envelope[i], envelope[i + 1])

    recalls_ext = [0.0] + list(recalls)
    envelope_ext = [envelope[0]] + envelope
    ap = 0.0
    for i in range(1, len(recalls_ext)):
        ap += (recalls_ext[i] - recalls_ext[i - 1]) * envelope_ext[i]
    return ap


def compute_detection_metrics(preds, gts, iou_thresh: float = 0.5) -> DetectionMatchResult:
    """preds: list of box tuples (x0,y0,x1,y1), already in confidence order
    (highest confidence first) -- this module doesn't take confidence
    scores directly, callers order preds themselves (matches this
    project's tiny hand-built fixtures, which have no real confidence
    gradation to sort by). gts: list of ground-truth box tuples for ONE
    object class."""
    precisions, recalls, tp, fp, fn = _precision_recall_at_each_rank(preds, gts, iou_thresh)
    ap50 = _ap_from_pr(precisions, recalls)
    precision = precisions[-1] if precisions else (1.0 if not gts else 0.0)
    recall = recalls[-1] if recalls else 0.0
    return DetectionMatchResult(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, ap50=ap50)


def compute_per_object_metrics(preds_by_object: dict, gts_by_object: dict, iou_thresh: float = 0.5) -> dict:
    """preds_by_object / gts_by_object: {object_id: [box, ...]}. Returns
    {object_id: {"precision":.., "recall":.., "ap50":..}} -- the exact
    input shape per_object_gate.evaluate_gate() expects."""
    result = {}
    for object_id, gts in gts_by_object.items():
        preds = preds_by_object.get(object_id, [])
        m = compute_detection_metrics(preds, gts, iou_thresh=iou_thresh)
        result[object_id] = {"precision": m.precision, "recall": m.recall, "ap50": m.ap50}
    return result
