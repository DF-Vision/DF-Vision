"""
checkpoint1_validation_scenes.py

Week 2, Day 1 deliverable -- Ground-truth validation (Checkpoint 1, hard gate)
scene design for the PRIMARY object (apple).

Per the master reference and Weeks 1-2 detailed plan:
  - Build 5-8 known-geometry validation scenes on apple, each with a
    HAND-CALCULABLE true visible fraction -- computed independently of the
    pipeline (marker pose + known occluder position/geometry), so this is
    a genuine ground-truth check, not a circular one.
  - Pass tolerance: MAE < 5 percentage points across all scenes. Hard gate
    -- nothing in Week 3+ proceeds until this passes (or failures are
    root-caused and fixed, per Day 6 buffer-day plan).
  - This script does NOT touch validation_scenes.csv directly (that file
    stays gated behind real marker placement, per RUNNING_NOTES.md Day 4
    finding: template rows are not evidence of a real Checkpoint 1 pass).
    This is the DESIGN stage -- occluder positions and the hand-calc
    methodology, ready to have real measured values dropped in once
    markers are physically mounted (per Week 2 Day 1 plan: marker
    mounting is finally allowed at this exact checkpoint).

STATUS: design skeleton. Object/occluder dimensions below use apple's
known approximate geometry (see APPLE_DIAMETER_CM) -- replace with the
ACTUAL measured apple + occluder dimensions once in hand, per the
marker-manifest ruler-check discipline already established
(apple marker = 3.5cm, confirmed against manifest.csv before trusting
any scene).

Hand-calc methodology (stated plainly, so it can be verified by a human
with a ruler and this formula, independent of any code):
  1. Apple is approximated as a circle in top-down projection (diameter
     = APPLE_DIAMETER_CM). This is an approximation -- a real apple is
     not a perfect sphere -- and this approximation error should be
     disclosed in the paper's Methods alongside the MAE result, not
     hidden.
  2. The occluder is a flat rectangular card of known width, placed at a
     known lateral offset and known height above the table (mirroring
     the MuJoCo occlusion pattern: occlusion in a top-down view requires
     the occluder to be ABOVE the target, not just beside it -- see
     RUNNING_NOTES.md Day 2, Bug 2).
  3. Visible fraction = 1 - (occluded circular-segment area / total
     circle area), where the occluded segment is computed from the
     lateral overlap between the occluder's projected footprint and the
     apple's projected circle.
  4. This is intentionally a SIMPLE, hand-verifiable model -- not a
     precise photometric simulation. The point of Checkpoint 1 is
     validating the PIPELINE's estimate against an independent
     calculation, not achieving perfect physical realism in the
     hand-calc itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


# ---------------------------------------------------------------------------
# Known geometry constants -- CONFIRM AGAINST REAL MEASUREMENTS BEFORE USE.
# These are placeholder/typical values. Replace with actual ruler
# measurements of the real apple and real occluder card before treating
# any scene's hand-calc as ground truth.
# ---------------------------------------------------------------------------

APPLE_DIAMETER_CM = 7.5          # PLACEHOLDER -- measure the real apple
APPLE_MARKER_SIZE_CM = 3.5       # per manifest.csv, already confirmed elsewhere
OCCLUDER_WIDTH_CM = 10.0         # PLACEHOLDER -- measure the real occluder card
OCCLUDER_HEIGHT_ABOVE_TABLE_CM = 5.0  # PLACEHOLDER -- must be > apple's own
                                        # height for genuine top-down occlusion,
                                        # mirroring the MuJoCo height_delta fix

MAE_TOLERANCE_PERCENTAGE_POINTS = 5.0  # hard gate threshold, per master reference


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ValidationSceneDesign:
    """
    Design for one Checkpoint 1 validation scene. `lateral_offset_cm` is
    the horizontal distance between the apple's center and the occluder's
    near edge, in the camera's top-down view -- this is the ONE parameter
    that varies across the sweep of scenes, mirroring the MuJoCo x_shift
    sweep design (Phase 0.25, Day 2-3).

    Positive lateral_offset_cm = occluder positioned further from the
    apple's center (less occlusion). 0.0 = occluder edge aligned with
    apple's center (maximum meaningful partial occlusion in this simple
    model).
    """
    scene_id: str
    lateral_offset_cm: float
    notes: str = ""

    # Populated by compute_hand_calculated_visible_fraction()
    hand_calculated_visible_fraction: float = field(init=False, default=0.0)

    # To be filled in once the pipeline is actually run against this real
    # scene (Week 2, Day 2). Left None until real data exists -- never
    # fabricate this value.
    pipeline_estimated_visible_fraction: float | None = field(init=False, default=None)


def compute_hand_calculated_visible_fraction(
        apple_diameter_cm: float,
        occluder_width_cm: float,
        lateral_offset_cm: float,
) -> float:
    """
    Computes the expected visible fraction of the apple's projected circle,
    given the occluder's lateral offset from the apple's center.

    This is the SAME formula a human should be able to work through by
    hand with a ruler and a calculator -- no pipeline code involved. This
    independence from the pipeline is what makes Checkpoint 1 a genuine
    ground-truth check rather than a circular one.

    Geometric model: the apple is a circle of radius r = diameter/2. The
    occluder is treated as an infinite (in this 2D top-down cross-section)
    straight edge at position `lateral_offset_cm` from the circle's
    center, extending inward by occluder_width_cm. The occluded region is
    the circular segment cut off by that edge.

    If lateral_offset_cm >= r: no occlusion (occluder doesn't reach the
    circle at all) -> visible fraction = 1.0.
    If lateral_offset_cm <= -r: full occlusion (occluder fully covers the
    circle) -> visible fraction = 0.0.
    Otherwise: partial occlusion via the standard circular segment area
    formula.
    """
    r = apple_diameter_cm / 2.0

    if lateral_offset_cm >= r:
        return 1.0
    if lateral_offset_cm <= -r:
        return 0.0

    # Standard circular segment area, where d = lateral_offset_cm is the
    # signed distance from the circle's center to the chord (occluder edge).
    d = lateral_offset_cm
    # Area of the circular segment beyond the chord at distance d from center:
    segment_area = (r ** 2) * math.acos(d / r) - d * math.sqrt(r ** 2 - d ** 2)
    circle_area = math.pi * (r ** 2)

    occluded_fraction = segment_area / circle_area
    visible_fraction = 1.0 - occluded_fraction

    return max(0.0, min(1.0, visible_fraction))


# ---------------------------------------------------------------------------
# The 6-scene design -- within the master reference's "5-8 scenes" range.
# Lateral offsets chosen to sweep from no-occlusion through full-occlusion,
# with denser sampling in the meaningful partial-occlusion zone (mirroring
# the lesson from Phase 0.25: the MuJoCo x_shift sweep saturated quickly,
# so this design deliberately spends more scenes in the region where
# visible_fraction is actually changing, not just at the extremes).
# ---------------------------------------------------------------------------

def build_scene_designs(apple_diameter_cm: float = APPLE_DIAMETER_CM) -> list[ValidationSceneDesign]:
    r = apple_diameter_cm / 2.0

    # Offsets as fractions of the radius, chosen to sample across the full
    # occlusion range with denser coverage in the middle (where a real
    # detector/TSDF pipeline is most likely to show interesting error).
    offset_fractions = [1.0, 0.5, 0.15, -0.15, -0.5, -1.0]

    scenes = []
    for i, frac in enumerate(offset_fractions):
        offset_cm = frac * r
        scene = ValidationSceneDesign(
            scene_id=f"apple_scene_{i}",
            lateral_offset_cm=round(offset_cm, 2),
            notes=(
                "no/minimal occlusion" if frac >= 1.0 else
                "full/near-full occlusion" if frac <= -1.0 else
                "partial occlusion"
            ),
        )
        scene.hand_calculated_visible_fraction = compute_hand_calculated_visible_fraction(
            apple_diameter_cm, OCCLUDER_WIDTH_CM, scene.lateral_offset_cm
        )
        scenes.append(scene)

    return scenes


# ---------------------------------------------------------------------------
# MAE computation -- run this ONCE real pipeline estimates exist (Week 2 Day 2)
# ---------------------------------------------------------------------------

def compute_mae(scenes: list[ValidationSceneDesign]) -> float | None:
    """
    Computes mean absolute error between hand-calculated and pipeline-
    estimated visible fractions, in percentage points.

    Returns None if any scene is missing a pipeline estimate -- this
    function refuses to silently compute a partial MAE, since that would
    misrepresent Checkpoint 1's actual completion status (same discipline
    as the validation_scenes.csv template-vs-real distinction).
    """
    missing = [s.scene_id for s in scenes if s.pipeline_estimated_visible_fraction is None]
    if missing:
        print(f"Cannot compute MAE -- pipeline estimates missing for: {missing}")
        return None

    errors = [
        abs(s.hand_calculated_visible_fraction - s.pipeline_estimated_visible_fraction) * 100.0
        for s in scenes
    ]
    return sum(errors) / len(errors)


def print_scene_design(scenes: list[ValidationSceneDesign]) -> None:
    print("=" * 78)
    print("Checkpoint 1 -- Apple Validation Scene Design")
    print("=" * 78)
    print(f"Apple diameter (cm): {APPLE_DIAMETER_CM}  <-- CONFIRM against real apple")
    print(f"Occluder width (cm): {OCCLUDER_WIDTH_CM}  <-- CONFIRM against real occluder")
    print(f"Occluder height above table (cm): {OCCLUDER_HEIGHT_ABOVE_TABLE_CM}  "
          f"<-- must exceed apple's own height for genuine top-down occlusion")
    print(f"MAE tolerance (hard gate): {MAE_TOLERANCE_PERCENTAGE_POINTS} percentage points")
    print()
    header = f"{'scene_id':>16} {'offset_cm':>10} {'hand_calc_vis':>14} {'notes':>22}"
    print(header)
    print("-" * len(header))
    for s in scenes:
        print(
            f"{s.scene_id:>16} {s.lateral_offset_cm:>10.2f} "
            f"{s.hand_calculated_visible_fraction:>14.4f} {s.notes:>22}"
        )
    print()
    print("NEXT STEPS (Week 2, Day 1-2, per master plan):")
    print("  1. Physically confirm apple + occluder dimensions with a ruler;")
    print("     update APPLE_DIAMETER_CM / OCCLUDER_WIDTH_CM above if they differ.")
    print("  2. Physically place apple + occluder at each scene's lateral_offset_cm,")
    print("     under the real oracle camera, markers mounted per the manifest.")
    print("  3. Run the REAL pipeline (detector + TSDF) against each real captured")
    print("     scene; record perceived_visibility as pipeline_estimated_visible_fraction.")
    print("  4. Call compute_mae() once all 6 scenes have real pipeline estimates.")
    print("  5. Pass tolerance: MAE < 5.0 percentage points. This is a HARD GATE --")
    print("     nothing in Week 3+ proceeds until this passes on apple in full,")
    print("     plus spot-checks on the other 4 objects (Week 2, Day 3).")


if __name__ == "__main__":
    scenes = build_scene_designs()
    print_scene_design(scenes)
