"""
auto_label_from_markers.py

C3 (project/closure session): the auto-labeling pipeline STATE.md flagged
as MISSING ("No marker-derived auto-labeling pipeline found yet"). Given
a captured RGB frame with a visible ArUco marker, produces a YOLO-format
label (class_id, normalized xc/yc/w/h) for the object that marker
identifies -- a dilated box around the marker's own detected corners, the
SAME convention `capture_and_gap.py`'s `build_bbox_mask_from_corners()`
already uses for its placeholder detection mask (train/serve parity:
wherever a marker stands in for a real detector box in this project, the
dilation formula must be identical, not reinvented per script).

class_id assignment comes from class_map.py (the single source of truth),
never a locally-defined list -- criterion (b) in
project_closure/CRITERIA.md's C3 section.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "real_camera"))
from class_map import load_class_index
from aruco_pose import build_camera_matrix  # noqa: F401 -- re-exported for callers' convenience

# Manifest-derived marker_id -> object_name (perception-calibration-gap/gazebo/markers/manifest.csv).
# Loaded once here rather than re-parsed per call.
MANIFEST_CSV = Path(__file__).parent.parent / "gazebo" / "markers" / "manifest.csv"


def load_marker_id_to_object() -> dict:
    import csv
    mapping = {}
    with open(MANIFEST_CSV) as f:
        for row in csv.DictReader(f):
            mapping[int(row["Marker ID"])] = row["Name"]
    return mapping


def dilated_bbox_from_marker_corners(marker_corners, image_shape, dilate_px: int = 60):
    """IDENTICAL formula to capture_and_gap.py's build_bbox_mask_from_corners
    (duplicated in pixel-rect form rather than mask form, since the labeler
    needs a rect for YOLO normalization, not a mask) -- x0,y0,x1,y1 pixel
    coords, clamped to image bounds."""
    h, w = image_shape[:2]
    pts = marker_corners[0].reshape(-1, 2)
    x_min, y_min = pts.min(axis=0) - dilate_px
    x_max, y_max = pts.max(axis=0) + dilate_px
    x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
    x_max, y_max = min(w, int(x_max)), min(h, int(y_max))
    return x_min, y_min, x_max, y_max


def bbox_to_yolo_line(class_id: int, bbox_px, image_shape) -> str:
    x0, y0, x1, y1 = bbox_px
    h, w = image_shape[:2]
    xc = ((x0 + x1) / 2.0) / w
    yc = ((y0 + y1) / 2.0) / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def yolo_line_to_bbox_px(line: str, image_shape):
    """Inverse of bbox_to_yolo_line -- decodes a written YOLO label line
    back into pixel coordinates, for round-trip verification."""
    parts = line.strip().split()
    class_id = int(parts[0])
    xc, yc, bw, bh = (float(v) for v in parts[1:5])
    h, w = image_shape[:2]
    x0 = (xc - bw / 2.0) * w
    x1 = (xc + bw / 2.0) * w
    y0 = (yc - bh / 2.0) * h
    y1 = (yc + bh / 2.0) * h
    return class_id, (x0, y0, x1, y1)


def auto_label_image(image_bgr: np.ndarray, dictionary=None, dilate_px: int = 60):
    """Detects every ArUco marker in image_bgr, maps each to an object via
    the manifest, and returns a list of YOLO label lines (only for markers
    whose Name appears in class_map.json -- e.g. occluder_A/world_origin
    markers are skipped, not mislabeled as a study object). Also returns
    the raw (class_id, bbox_px) pairs for tests/plotting."""
    if dictionary is None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    marker_to_object = load_marker_id_to_object()
    class_index = load_class_index()

    lines, raw = [], []
    if ids is None:
        return lines, raw
    for i, marker_id in enumerate(ids.flatten()):
        obj_name = marker_to_object.get(int(marker_id))
        if obj_name is None or obj_name not in class_index:
            continue  # not a study object's marker (e.g. occluder/world-origin) -- skip, don't mislabel
        class_id = class_index[obj_name]
        bbox_px = dilated_bbox_from_marker_corners(corners[i], image_bgr.shape, dilate_px=dilate_px)
        lines.append(bbox_to_yolo_line(class_id, bbox_px, image_bgr.shape))
        raw.append((class_id, obj_name, bbox_px))
    return lines, raw


def draw_contact_sheet(images_and_raw, out_path):
    """SYNTHETIC contact sheet: draws each image's auto-labeled boxes for
    visual inspection. images_and_raw: list of (image_bgr, raw) pairs."""
    cells = []
    for img, raw in images_and_raw:
        canvas = img.copy()
        for class_id, obj_name, (x0, y0, x1, y1) in raw:
            cv2.rectangle(canvas, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
            cv2.putText(canvas, f"{obj_name}({class_id})", (int(x0), max(0, int(y0) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cells.append(canvas)
    if not cells:
        return
    h, w = cells[0].shape[:2]
    n = len(cells)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    sheet = np.full((rows * h, cols * w, 3), 40, dtype=np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = cell
    cv2.imwrite(str(out_path), sheet)
