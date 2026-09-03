"""
daily_integration_check.py

Day 7 deliverable (Week 1 plan). Loads both people's latest code and runs
ONE synthetic trial end-to-end through the MERGED pipeline:

    Gap computation (real gap_computer.py, real occlusion_estimator.py,
    real tsdf_fusion.py's synthetic self-test frame generator)
        -> stub compute_prg_edge_error() (Day 5 stub, mock-pipeline/)
        -> merged nonconformity score (real conformal_gate.py's scoring
           logic, evaluated inline here since this is a single synthetic
           trial, not a calibration/test split)
        -> stub explain() (Day 6 stub, exploratory_day6/)
        -> placeholder voice/TTS field (string placeholder only -- no
           audio call here, that's Day 6's separate smoke test)

...and asserts every schema field in logger.py's REQUIRED_FIELDS plus the
anticipated collaborator fields populates correctly.

WHAT THIS PROVES: the plumbing between every module -- real and stub --
is wired correctly and schema-compatible, BEFORE the collaborator's real
code exists. It does NOT prove anything about Gap/PRG_edge_error's actual
predictive value (that's C1/C4's job, on real frozen data) -- this is a
pipeline-wiring check only, same status as the MuJoCo Phase 0.25 smoke
test: never cited as evidence for any statistical result.

ONCE THE COLLABORATOR JOINS: her real compute_prg_edge_error() and
explain() replace the two stub imports below (see the two import lines
marked STUB SWAP POINT) -- nothing else in this script should need to
change, since the interface contracts (SCHEMA.md Section 4, and whatever
explain() signature is agreed) are fixed independent of implementation.

Run from the project root (perception-calibration-gap/):
    python3 daily_integration_check.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

# Matches run_trial.py's own established sys.path.insert() pattern exactly.
sys.path.insert(0, str(Path(__file__).parent / "gazebo"))
sys.path.insert(0, str(Path(__file__).parent / "mock-pipeline"))
sys.path.insert(0, str(Path(__file__).parent / "exploratory_day6"))

from occlusion_estimator import ObjectGeometry, Pose, compute_true_visibility
from gap_computer import compute_gap_features
from tsdf_fusion import MaskedTSDFFusion, CameraIntrinsics, make_synthetic_depth_frame
from conformal_gate import ConformalGate
from logger import TrialRecord, TrialLogger

# --- STUB SWAP POINT 1: replace with collaborator's real implementation ---
from prg_edge_error_stub import compute_prg_edge_error

# --- STUB SWAP POINT 2: replace with collaborator's real implementation ---
from explain_stub import explain


LOG_PATH = Path(__file__).parent / "mock-pipeline" / "data" / "trials_daily_integration.jsonl"


def build_synthetic_trial(seed: int = 0):
    """One fully synthetic trial: real Gap computation via real modules,
    on a fabricated (not MuJoCo, not real-hardware) scene, exactly mirroring
    tsdf_fusion.py's and occlusion_estimator.py's own self-test conventions.
    This keeps this script self-contained and independent of the MuJoCo
    container / real hardware, since it's meant to run daily, cheaply, and
    from either person's machine."""
    rng = np.random.default_rng(seed)

    intr = CameraIntrinsics(width=640, height=480, fx=500, fy=500, cx=320, cy=240)
    target_center = np.array([0.0, 0.0, 0.5])
    target_half = np.array([0.04, 0.04, 0.04])

    # perceived_visibility via real tsdf_fusion.py, on a synthetic frame
    # with a partial occluder (mirrors tsdf_fusion.py's own self-test Case 4).
    occluder_box = (320, 0, 400, 480)  # partial overlap only, not full target bbox
    color, depth, mask = make_synthetic_depth_frame(
        640, 480, intr, target_center, target_half,
        occluder_uv_box=occluder_box, seed=seed,
    )
    fusion = MaskedTSDFFusion(voxel_length=0.004, sdf_trunc=0.02)
    fusion.integrate_frame(color, depth, mask, intr)
    perceived_visibility = fusion.compute_perceived_visibility(
        target_center, target_half, grid_resolution=8
    )

    # true_visibility via real occlusion_estimator.py, independent scene
    # description (per that module's own self-test Case 4: partial occlusion).
    target_pose = Pose(x=0.0, y=0.0, z=0.5)
    target_geom = ObjectGeometry(width=0.08, height=0.08, depth=0.08)
    occluder_pose = Pose(x=0.045, y=0.0, z=0.49)
    occluder_geom = ObjectGeometry(width=0.08, height=0.08, depth=0.02)
    intrinsics_dict = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
    true_visibility = compute_true_visibility(
        target_pose, target_geom, occluder_pose, occluder_geom, intrinsics_dict
    )

    # Gap via real gap_computer.py, single-step trajectory.
    gap_result = compute_gap_features([perceived_visibility], [true_visibility])

    return perceived_visibility, true_visibility, gap_result, rng


def main():
    print("=== Daily Integration Check ===\n")

    if LOG_PATH.exists():
        LOG_PATH.unlink()
        print(f"[reset] removed stale integration-check log at {LOG_PATH}")
    logger = TrialLogger(str(LOG_PATH))

    perceived_visibility, true_visibility, gap_result, rng = build_synthetic_trial(seed=0)
    print(f"[real] perceived_visibility={perceived_visibility:.4f}  "
          f"(tsdf_fusion.py, unmodified)")
    print(f"[real] true_visibility={true_visibility:.4f}  "
          f"(occlusion_estimator.py, unmodified)")
    print(f"[real] Gap_initial={gap_result.gap_initial:.4f}  "
          f"(gap_computer.py, unmodified)")

    # --- STUB: PRG edges (SCHEMA.md Section 3a/3b shape: list of per-edge dicts) ---
    prg_edges = compute_prg_edge_error(rgb=None, depth=None, target_bbox=None, rng=rng)
    prg_value = prg_edges[0]["prg_edge_error"]  # single synthetic edge here -- only one to feed the gate score below
    print(f"[STUB] prg_edges={prg_edges}  (placeholder -- never a real finding)")

    # --- Merged nonconformity score (real ConformalGate scoring math, applied
    # inline to this single synthetic trial -- NOT a real calibration/fit,
    # since that requires a calibration set; this only proves the max-
    # combination scoring formula from PREREG.md C1-ext/C4-ext runs without
    # error against both a real Gap value and a stub PRG value together) ---
    # Using placeholder scale denominators (=1.0) since there's no real
    # calibration set here -- this is a WIRING check, not a fitted gate.
    gap_score = gap_result.gap_initial / 1.0
    prg_score = prg_value / 1.0
    merged_nonconformity_score = max(gap_score, prg_score)
    dominant_signal = "gap" if gap_score >= prg_score else "prg"
    print(f"[merged] nonconformity_score={merged_nonconformity_score:.4f}  "
          f"dominant_deferral_signal={dominant_signal!r}  "
          f"(max-combination formula per PREREG.md C4-ext, unfitted/placeholder scale)")

    # --- STUB: explain() ---
    explanation = explain({"dominant_deferral_signal": dominant_signal, "object_id": "apple"})
    print(f"[STUB] explain() -> {explanation!r}")

    # --- Placeholder voice field (no real audio call in this script) ---
    voice_command_text_placeholder = None  # ASR is collaborator's real component

    record = TrialRecord(
        trial_id=0, session_id=0, object_id="apple",
        session_baseline_failure_rate=0.0,
        measured_true_visibility=true_visibility,
        true_occlusion=1.0 - true_visibility,
        clutter_count=1,
        perceived_visibility_initial=perceived_visibility,
        Gap_initial=gap_result.gap_initial,
        Gap_max=gap_result.gap_max,
        Gap_slope=gap_result.gap_slope,
        failure=0,
        camera_pose_variant="synthetic_daily_check",
        nonconformity_score=merged_nonconformity_score,
        gate_fired=False,  # placeholder scale, not a fitted threshold -- never a real gate decision
        gate_warmup_phase=True,
        prg_edges=prg_edges,
        prg_is_stub=True,
        prg_error_note=None,
        dominant_deferral_signal=dominant_signal,
        voice_command_text=voice_command_text_placeholder,
        grounding_confidence=None,
    )
    logger.append_trial(record)
    print("\n[logger] trial appended successfully")

    loaded = logger.load_trials()
    assert len(loaded) == 1, "Expected exactly one trial in the integration-check log"
    trial = loaded[0]

    # --- Schema completeness check ---
    from dataclasses import fields
    # Fields legitimately allowed to stay None in this wiring-only check
    # (mirrors mujoco_day3.py's own allowed_none convention).
    allowed_none = {
        "policy_checkpoint_hash", "policy_seed", "Gap_before_fusion",
        "Gap_after_fusion", "outcome", "failure_type", "goal_pose_tolerance",
        "camera_pose", "light_level", "seed", "gap_per_step", "timing_per_step",
        "true_visibility_per_step", "tsdf_fusion_attempted",
        "voice_command_text", "grounding_confidence", "prg_error_note",
    }
    bad_fields = []
    for f in fields(record):
        if trial.get(f.name) is None and f.name not in allowed_none:
            bad_fields.append(f.name)

    print(f"\nSchema fields with unexpectedly-None values: {bad_fields or 'none'}")

    gate_pass = (len(loaded) == 1) and (not bad_fields)
    print(f"\n=== Daily Integration Check: {'PASS' if gate_pass else 'FAIL -- see above'} ===")
    if gate_pass:
        print("Both real modules (gap_computer, occlusion_estimator, tsdf_fusion, "
              "logger) and both Day 5/6 stubs (prg_edge_error, explain) are wired "
              "correctly and schema-compatible.")
        print(f"Log written to: {LOG_PATH}")
        print("\nReminder: this trial's numbers are NEVER real findings -- "
              "synthetic scene + stub PRG_edge_error + unfitted placeholder-scale "
              "gate. Wiring confirmation only, same status as the MuJoCo Phase "
              "0.25 smoke test.")


if __name__ == "__main__":
    main()
