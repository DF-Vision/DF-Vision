"""
logger.py

Purpose
-------
Append-only JSONL trial logger. One JSON object per line = one trial record.

Schema
------
REQUIRED_FIELDS (unchanged from the original canonical version -- these are
the fields every trial record MUST have, validated on every append_trial()
call):
    trial_id, session_id, object_id, session_baseline_failure_rate,
    measured_true_visibility, true_occlusion, clutter_count,
    perceived_visibility_initial, Gap_initial, Gap_max, Gap_slope,
    failure, camera_pose_variant

EXTENSION (new, optional fields -- added to support C4/C5's pre-registered
metrics, which the original schema had no room for):
    nonconformity_score, gate_fired, gate_warmup_phase   -- C4 online gate state
    tsdf_fusion_attempted, Gap_before_fusion, Gap_after_fusion -- C5 recovery tracking
    outcome, failure_type                                -- richer than the bare
                                                             0/1 `failure` flag
    goal_pose_tolerance, camera_pose, light_level, seed  -- scenario parameters,
                                                             useful for debugging
                                                             and exchangeability
                                                             checks (KS test across
                                                             sessions/objects)
    gap_per_step, timing_per_step, true_visibility_per_step -- full trajectories,
                                                             for anything that needs
                                                             more than the 3 summary
                                                             Gap statistics

These are ALL optional (default None) specifically so:
  (a) REQUIRED_FIELDS and the missing-field validation logic are unchanged,
  (b) the existing self-tests below (round-trip, freeze-blocks-append,
      missing-field validation) still pass unmodified,
  (c) any other script that constructs a bare-minimum TrialRecord (only the
      originally-required fields) keeps working exactly as before.

Edge cases handled (unchanged from original):
  - Append mode only.
  - Malformed/partial last line skippable via load_trials(strict=False).
  - Freeze discipline: once FREEZE_MARKER_PATH exists, append_trial() raises.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional, List

REQUIRED_FIELDS = [
    "trial_id", "session_id", "object_id", "session_baseline_failure_rate",
    "measured_true_visibility", "true_occlusion", "clutter_count",
    "perceived_visibility_initial", "Gap_initial", "Gap_max", "Gap_slope",
    "failure", "camera_pose_variant",
]


@dataclass
class TrialRecord:
    trial_id: int
    session_id: int
    object_id: str
    session_baseline_failure_rate: float
    measured_true_visibility: float
    true_occlusion: float
    clutter_count: int
    perceived_visibility_initial: float
    Gap_initial: float
    Gap_max: float
    Gap_slope: float
    failure: int
    camera_pose_variant: str
    policy_checkpoint_hash: Optional[str] = None  # renamed from openvla_checkpoint_hash --
                                                    # OpenVLA dropped; scripted/IK policy still
                                                    # needs a reproducibility identifier (e.g. a
                                                    # git commit hash of execute_grasp.py + IK
                                                    # params, pinned once per Week1 protocol-lock)
    policy_seed: Optional[int] = None              # renamed from openvla_seed
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # --- EXTENSION: optional fields for C4 (online gate) / C5 (fusion recovery) ---
    nonconformity_score: Optional[float] = None
    gate_fired: Optional[bool] = None
    gate_warmup_phase: Optional[bool] = None
    tsdf_fusion_attempted: Optional[bool] = None
    Gap_before_fusion: Optional[float] = None
    Gap_after_fusion: Optional[float] = None
    outcome: Optional[str] = None
    failure_type: Optional[str] = None
    goal_pose_tolerance: Optional[float] = None
    camera_pose: Optional[List[float]] = None
    light_level: Optional[float] = None
    seed: Optional[int] = None
    gap_per_step: Optional[List[float]] = None
    timing_per_step: Optional[List[float]] = None
    true_visibility_per_step: Optional[List[float]] = None

    # --- EXTENSION: collaborator's real interface, per SCHEMA.md Revision 3 Section 3a/3b ---

    # prg_edge_error (single scalar) replaced -- a trial's target can have >1 adjacent
    # object; collaborator's own recommendation was to log all target-adjacent pairs,
    # not pre-select one "the" neighbor.

    prg_edges: Optional[List[dict]] = None      # list of per-edge dicts (Section 3b shape),
                                            # or [] if target has no adjacent object this
                                            # frame -- valid, not an error

    prg_is_stub: Optional[bool] = None          # True if prg_edges came from the seeded stub
                                            # (config flag, missing camera_intrinsics, or an
                                            # exception in the real chain)

    prg_error_note: Optional[str] = None        # populated only if the real chain raised

    dominant_deferral_signal: Optional[str] = None
    voice_command_text: Optional[str] = None
    grounding_confidence: Optional[float] = None


class TrialLogger:
    def __init__(self, log_path: str, freeze_marker_path: Optional[str] = None):
        self.log_path = log_path
        self.freeze_marker_path = freeze_marker_path or (log_path + ".FROZEN")
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def is_frozen(self) -> bool:
        return os.path.exists(self.freeze_marker_path)

    def freeze(self):
        with open(self.freeze_marker_path, "w") as f:
            f.write(f"Frozen at {datetime.now(timezone.utc).isoformat()}\n")

    def append_trial(self, record: TrialRecord):
        if self.is_frozen():
            raise RuntimeError(
                f"Dataset is frozen ({self.freeze_marker_path} exists). "
                f"Refusing to append trial_id={record.trial_id}."
            )
        missing = [f for f in REQUIRED_FIELDS if getattr(record, f, None) is None]
        if missing:
            raise ValueError(f"TrialRecord missing required fields: {missing}")

        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def load_trials(self, strict: bool = True) -> List[dict]:
        if not os.path.exists(self.log_path):
            return []
        trials = []
        with open(self.log_path) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                trials.append(json.loads(line))
            except json.JSONDecodeError:
                if strict:
                    raise
                is_last = (i == len(lines) - 1)
                if not is_last:
                    raise
        return trials


if __name__ == "__main__":
    test_log_path = "mock-pipeline/data/test_trials.jsonl"
    if os.path.exists(test_log_path):
        os.remove(test_log_path)
    freeze_marker = test_log_path + ".FROZEN"
    if os.path.exists(freeze_marker):
        os.remove(freeze_marker)

    logger = TrialLogger(test_log_path)

    r1 = TrialRecord(
        trial_id=0, session_id=0, object_id="apple",
        session_baseline_failure_rate=0.05, measured_true_visibility=0.7,
        true_occlusion=0.3, clutter_count=2, perceived_visibility_initial=0.75,
        Gap_initial=0.05, Gap_max=0.08, Gap_slope=0.01, failure=0,
        camera_pose_variant="front_30deg",
    )
    logger.append_trial(r1)
    print("Appended trial 0 OK")

    trials = logger.load_trials()
    assert len(trials) == 1 and trials[0]["object_id"] == "apple"
    print("Load-back OK:", trials[0])

    logger.freeze()
    try:
        logger.append_trial(r1)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        print("Freeze discipline correctly enforced:", e)

    logger2 = TrialLogger("mock-pipeline/data/test_trials2.jsonl")
    bad = TrialRecord(
        trial_id=1, session_id=0, object_id="apple",
        session_baseline_failure_rate=0.05, measured_true_visibility=0.7,
        true_occlusion=0.3, clutter_count=2, perceived_visibility_initial=0.75,
        Gap_initial=0.05, Gap_max=0.08, Gap_slope=0.01, failure=0,
        camera_pose_variant=None,
    )
    try:
        logger2.append_trial(bad)
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        print("Missing-field validation correctly enforced:", e)

    # New: confirm the EXTENSION fields round-trip correctly and don't break
    # required-field validation when left unset.
    r2 = TrialRecord(
        trial_id=2, session_id=0, object_id="glass",
        session_baseline_failure_rate=0.10, measured_true_visibility=0.4,
        true_occlusion=0.6, clutter_count=1, perceived_visibility_initial=0.55,
        Gap_initial=0.15, Gap_max=0.22, Gap_slope=0.02, failure=1,
        camera_pose_variant="front_0deg",
        nonconformity_score=1.8, gate_fired=True, gate_warmup_phase=False,
        tsdf_fusion_attempted=True, Gap_before_fusion=0.22, Gap_after_fusion=0.09,
        outcome="failure", failure_type="grasp_slip",
    )
    logger3 = TrialLogger("mock-pipeline/data/test_trials3.jsonl")
    logger3.append_trial(r2)
    loaded = logger3.load_trials()
    assert loaded[0]["gate_fired"] is True
    assert loaded[0]["Gap_after_fusion"] == 0.09
    print("Extension fields round-trip OK:", loaded[0])

    # New: confirm prg_edges (list[dict]) round-trips correctly through JSON,
    # and that a stub trial vs. a real-chain trial are distinguishable on load.

    r3 = TrialRecord(
        trial_id=3,
        session_id=0,
        object_id="purse",
        session_baseline_failure_rate=0.05,
        measured_true_visibility=0.6,
        true_occlusion=0.4,
        clutter_count=1,
        perceived_visibility_initial=0.5,
        Gap_initial=0.1,
        Gap_max=0.1,
        Gap_slope=0.0,
        failure=0,
        camera_pose_variant="front_0deg",

        prg_edges=[{
            "neighbor_name": "glass",
            "relation_prediction": "Contact",
            "relation_confidence": 0.82,
            "prg_edge_error": 0.23,
            "geometric_feature_vector": {
                "boundary_sharpness": 0.4
            },
            "mask_is_bbox_placeholder": True,
        }],
        prg_is_stub=False,
        prg_error_note=None,
    )

    logger4 = TrialLogger("mock-pipeline/data/test_trials4.jsonl")
    logger4.append_trial(r3)

    loaded4 = logger4.load_trials()

    assert loaded4[0]["prg_edges"][0]["neighbor_name"] == "glass"
    assert loaded4[0]["prg_is_stub"] is False

    print("prg_edges round-trip OK:", loaded4[0]["prg_edges"])


    print("\nAll logger.py self-tests passed.")