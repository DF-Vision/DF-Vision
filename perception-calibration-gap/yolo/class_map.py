"""
class_map.py

C3: loads the single source of truth for object_name -> class index
(class_map.json), used by train_yolo_scaffold.py, auto_label_from_markers.py,
and detection_metrics.py's tests. No script defines its own copy of the
class list.
"""
import json
from pathlib import Path

_CLASS_MAP_JSON = Path(__file__).parent / "class_map.json"


def load_class_names() -> list:
    with open(_CLASS_MAP_JSON) as f:
        data = json.load(f)
    return list(data["classes"])


def load_class_index() -> dict:
    """object_name -> class_id"""
    return {name: i for i, name in enumerate(load_class_names())}
