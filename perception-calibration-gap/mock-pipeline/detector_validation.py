"""
detector_validation.py

Week 1, Day 6 deliverable — Open-vocabulary detector localization-quality
validation, per object, against marker-based ground truth.

Per the master reference (Section 3a) and Weeks 1-2 detailed plan:
  - The detector (YOLO-World / Grounding DINO, zero-shot) is infrastructure,
    not a claimed contribution -- but its bbox quality directly propagates
    into TSDF-masked perceived_visibility, so it must be validated per
    object before being trusted in real trials.
  - This is a PER-OBJECT gate, not an aggregate one. An aggregate detection
    metric can hide one object (most likely candidate: glass, given
    transparency) where the detector is quietly bad.
  - Any object that fails triggers the documented fallback: short 30-60s
    video capture -> ffmpeg frame extraction -> auto-label via the
    already-validated marker pipeline -> fine-tune YOLOv8n on that object
    only, from pretrained COCO weights.

STATUS: skeleton only. Real detector calls and real marker-based ground
truth are NOT yet wired in -- this file defines the structure, the pass
criteria, and the fallback trigger logic so real data can be dropped in
once objects are physically available (Week 1, Day 6 per the hardware
timeline).

Nothing in this file should be run against fabricated/placeholder scene
data and reported as a real validation result -- same discipline as
validation_scenes.csv vs. validation_scenes_template.csv (see
RUNNING_NOTES.md, Day 4 finding: template rows are not evidence of a
real Checkpoint 1 pass).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import json
import datetime


# ---------------------------------------------------------------------------
# Study constants -- must match PREREG.md / SCHEMA.md exactly. Do not diverge.
# ---------------------------------------------------------------------------

STUDY_OBJECTS = ["apple", "tennis_ball", "purse", "glass", "stapler"]

# Per-object IoU acceptance threshold. A single global threshold is used for
# now; if a per-object threshold is later justified (e.g. glass needs a wider
# tolerance due to known transparency/reflectivity difficulty), that decision
# must be dated and stated explicitly in the paper's Limitations section,
# not silently baked in here.
IOU_ACCEPTANCE_THRESHOLD = 0.5  # placeholder -- confirm against mentor/paper
                                 # convention before real validation runs

# Objects expected, a priori, to be the hardest cases (per master reference
# Section 2 -- glass is the deliberate hard case for detection + depth).
# This is NOT a lowered bar for these objects -- it's a flag for extra
# attention and more careful root-cause analysis if they fail.
EXPECTED_DIFFICULT_OBJECTS = {"glass"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates: (x_min, y_min, x_max, y_max)."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def area(self) -> float:
        w = max(0.0, self.x_max - self.x_min)
        h = max(0.0, self.y_max - self.y_min)
        return w * h


@dataclass
class ValidationScene:
    """
    A single real captured scene used to validate the detector against
    marker-based ground truth for one object.

    STATUS: fields defined, but this must be populated from REAL captured
    images and REAL marker-derived ground truth boxes -- never from
    synthetic/placeholder data treated as if it were real.
    """
    scene_id: str
    object_id: str                  # must be one of STUDY_OBJECTS
    image_path: str                 # path to the real captured RGB frame
    ground_truth_bbox: BBox         # derived from marker pose + known geometry,
                                     # independent of any detector
    detector_bbox: Optional[BBox] = None  # populated by run_detector_on_scene()
    detector_confidence: Optional[float] = None
    notes: str = ""


@dataclass
class ObjectValidationResult:
    object_id: str
    scenes: list[ValidationScene] = field(default_factory=list)
    per_scene_iou: list[float] = field(default_factory=list)
    mean_iou: Optional[float] = None
    passed: Optional[bool] = None
    fallback_triggered: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Core IoU computation
# ---------------------------------------------------------------------------

def compute_bbox_iou(box_a: BBox, box_b: BBox) -> float:
    """
    Standard axis-aligned bounding box IoU (Intersection over Union).

    Returns 0.0 if there is no overlap, including the degenerate case where
    either box has zero area (never divides by zero silently).
    """
    inter_x_min = max(box_a.x_min, box_b.x_min)
    inter_y_min = max(box_a.y_min, box_b.y_min)
    inter_x_max = min(box_a.x_max, box_b.x_max)
    inter_y_max = min(box_a.y_max, box_b.y_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    intersection = inter_w * inter_h

    union = box_a.area() + box_b.area() - intersection
    if union <= 0.0:
        return 0.0

    return intersection / union


# ---------------------------------------------------------------------------
# Detector call -- STUB. Replace with real open-vocabulary detector call
# (YOLO-World / Grounding DINO), text-prompted per trial with object_name,
# per Section 3a of the master reference.
# ---------------------------------------------------------------------------

def run_detector_on_scene(scene: ValidationScene,
                           detector_fn: Optional[Callable[[str, str], BBox]] = None
                           ) -> BBox:
    """
    Runs the real open-vocabulary detector on `scene.image_path`, text-prompted
    with `scene.object_id`, and returns the resulting bounding box.

    detector_fn: injected callable (image_path, object_name) -> BBox, so this
    function can be tested independently of whichever real detector library
    ends up wired in. If None, raises -- this function must never silently
    fabricate a bbox.
    """
    if detector_fn is None:
        raise NotImplementedError(
            "run_detector_on_scene() has no real detector wired in yet. "
            "Pass a detector_fn, or wire the real open-vocabulary detector "
            "call here once available. Do NOT stub this with a fabricated "
            "bbox -- that would silently corrupt validation results."
        )
    return detector_fn(scene.image_path, scene.object_id)


# ---------------------------------------------------------------------------
# Per-object validation
# ---------------------------------------------------------------------------

def validate_object(object_id: str,
                     scenes: list[ValidationScene],
                     detector_fn: Optional[Callable[[str, str], BBox]] = None,
                     iou_threshold: float = IOU_ACCEPTANCE_THRESHOLD
                     ) -> ObjectValidationResult:
    """
    Runs the detector against every scene for a single object, computes
    per-scene IoU against marker-based ground truth, and gates pass/fail
    for THIS OBJECT ALONE.

    This function must be called once per object, separately -- never
    pooled across objects before gating. An aggregate metric can hide one
    object where the detector is quietly bad (per Week 1 Day 6 plan).
    """
    if object_id not in STUDY_OBJECTS:
        raise ValueError(
            f"object_id={object_id!r} is not one of the five pre-registered "
            f"study objects {STUDY_OBJECTS}. Do not validate against an "
            f"object outside the pre-registered set without a dated "
            f"PREREG.md amendment."
        )

    result = ObjectValidationResult(object_id=object_id, scenes=scenes)

    for scene in scenes:
        if scene.object_id != object_id:
            raise ValueError(
                f"Scene {scene.scene_id!r} has object_id={scene.object_id!r}, "
                f"expected {object_id!r}. Mixed-object scene lists indicate "
                f"a bug in how scenes were assembled, not a detector issue."
            )

        detector_bbox = run_detector_on_scene(scene, detector_fn=detector_fn)
        scene.detector_bbox = detector_bbox

        iou = compute_bbox_iou(detector_bbox, scene.ground_truth_bbox)
        result.per_scene_iou.append(iou)

    if result.per_scene_iou:
        result.mean_iou = sum(result.per_scene_iou) / len(result.per_scene_iou)
        result.passed = result.mean_iou >= iou_threshold
    else:
        result.mean_iou = None
        result.passed = False
        result.notes = "No scenes provided -- cannot validate, treat as fail."

    if object_id in EXPECTED_DIFFICULT_OBJECTS:
        result.notes += (
            f" [flagged a priori as expected-difficult per master reference "
            f"Section 2; a fail here is expected-difficulty territory, not "
            f"necessarily a pipeline bug -- but still must be resolved or "
            f"explicitly disclosed as a per-object limitation.]"
        )

    return result


# ---------------------------------------------------------------------------
# Cross-object confusion check (separate from per-object IoU gate)
# ---------------------------------------------------------------------------

def check_cross_object_confusion(detections: list[tuple[str, str]]) -> dict:
    """
    detections: list of (true_object_id, detector_predicted_label) pairs,
    collected across ALL objects' validation scenes.

    Returns a simple confusion count dict: {(true, predicted): count}.

    Per the master reference: apple/tennis_ball is a DELIBERATE visual-
    similarity stress test. Some confusion between this pair is expected
    and not itself a failure -- SEVERE confusion is. This function reports
    the raw counts; judging "severe" is a documented, human decision, not
    something this function should silently threshold.
    """
    confusion: dict[tuple[str, str], int] = {}
    for true_id, predicted_label in detections:
        key = (true_id, predicted_label)
        confusion[key] = confusion.get(key, 0) + 1
    return confusion


# ---------------------------------------------------------------------------
# Fallback trigger -- documented fine-tune path
# ---------------------------------------------------------------------------

@dataclass
class FallbackPlan:
    object_id: str
    reason: str
    video_capture_done: bool = False
    frames_extracted: bool = False
    auto_labeled: bool = False
    fine_tune_complete: bool = False
    fine_tune_acceptance_passed: Optional[bool] = None
    dated: str = field(default_factory=lambda: datetime.date.today().isoformat())


def trigger_fallback(object_id: str, mean_iou: float, threshold: float) -> FallbackPlan:
    """
    Called when validate_object() returns passed=False for a given object.

    Documents the fallback per Section 3a of the master reference:
      1. Capture a short 30-60s video across pose/occlusion/lighting variation.
      2. Extract frames via ffmpeg at a few frames per second.
      3. Auto-label via the already-validated marker-based ground-truth pipeline
         (never an unvalidated pipeline -- see Week 1/2 edge cases).
      4. Fine-tune YOLOv8n from pretrained COCO weights on this object only.
      5. Re-run this SAME per-object IoU gate on the fine-tuned model before
         trusting it in real trials.

    This function only creates and returns the plan record -- it does not
    itself run ffmpeg, labeling, or training. Those are separate, explicit
    steps once real hardware/video exists.
    """
    plan = FallbackPlan(
        object_id=object_id,
        reason=(
            f"Zero-shot detector mean IoU={mean_iou:.3f} fell below "
            f"acceptance threshold={threshold:.3f} for object_id={object_id!r}."
        ),
    )
    return plan


# ---------------------------------------------------------------------------
# Top-level orchestration -- run this once real scenes exist for all 5 objects
# ---------------------------------------------------------------------------

def run_full_detector_validation(
        scenes_by_object: dict[str, list[ValidationScene]],
        detector_fn: Optional[Callable[[str, str], BBox]] = None,
        iou_threshold: float = IOU_ACCEPTANCE_THRESHOLD,
) -> dict[str, ObjectValidationResult]:
    """
    Runs validate_object() independently for every object in STUDY_OBJECTS.

    Raises if any object in STUDY_OBJECTS is missing from scenes_by_object --
    this validation is required for every object individually before real
    trial collection starts, per the master reference's acceptance checklist.
    """
    missing = set(STUDY_OBJECTS) - set(scenes_by_object.keys())
    if missing:
        raise ValueError(
            f"Missing validation scenes for objects: {sorted(missing)}. "
            f"All five study objects require this check individually -- "
            f"do not proceed to real trial collection with any object "
            f"ungated."
        )

    results: dict[str, ObjectValidationResult] = {}
    for object_id in STUDY_OBJECTS:
        results[object_id] = validate_object(
            object_id,
            scenes_by_object[object_id],
            detector_fn=detector_fn,
            iou_threshold=iou_threshold,
        )

    return results


def summarize_results(results: dict[str, ObjectValidationResult]) -> str:
    """Human-readable summary table -- for the running notes / paper draft."""
    lines = [
        "=== Detector Validation Summary (per-object, NOT aggregate) ===",
    ]
    all_passed = True
    for object_id in STUDY_OBJECTS:
        r = results.get(object_id)
        if r is None:
            lines.append(f"  {object_id:12s}  NO DATA")
            all_passed = False
            continue
        status = "PASS" if r.passed else "FAIL"
        if not r.passed:
            all_passed = False
        mean_iou_str = f"{r.mean_iou:.3f}" if r.mean_iou is not None else "N/A"
        lines.append(
            f"  {object_id:12s}  mean_IoU={mean_iou_str}  {status}"
            f"{'  <-- fallback needed' if not r.passed else ''}"
        )
        if r.notes:
            lines.append(f"      note: {r.notes.strip()}")

    lines.append("")
    lines.append(
        "Overall: ALL OBJECTS PASSED" if all_passed
        else "Overall: NOT ALL OBJECTS PASSED -- see fallback triggers above"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test (synthetic data only -- NEVER treat this as a real validation run)
# ---------------------------------------------------------------------------

def _self_test():
    """
    Exercises the code paths above with fabricated boxes, purely to confirm
    the plumbing works. This produces NO evidence about real detector
    quality and must never be cited as a validation result -- same
    discipline as the MuJoCo Phase 0.25 smoke test.
    """
    print("Running detector_validation.py self-test (synthetic data only)...")

    def fake_detector(image_path: str, object_name: str) -> BBox:
        # Deliberately offset from a nominal "ground truth" box below,
        # to produce a non-trivial (not 0, not 1) IoU.
        return BBox(100, 100, 200, 200)

    scenes_by_object: dict[str, list[ValidationScene]] = {}
    for i, object_id in enumerate(STUDY_OBJECTS):
        # Ground truth box shifts slightly per object so IoUs differ.
        gt = BBox(90 + i * 5, 90, 190 + i * 5, 190)
        scenes_by_object[object_id] = [
            ValidationScene(
                scene_id=f"{object_id}_synth_0",
                object_id=object_id,
                image_path="SYNTHETIC_SELF_TEST_ONLY.png",
                ground_truth_bbox=gt,
            )
        ]

    results = run_full_detector_validation(
        scenes_by_object, detector_fn=fake_detector, iou_threshold=IOU_ACCEPTANCE_THRESHOLD
    )
    print(summarize_results(results))
    print("\nSelf-test complete. Reminder: NONE of the above is real evidence.")


if __name__ == "__main__":
    _self_test()