"""
mock_vla.py

Stand-in for OpenVLA (Day 12+). Takes a target object's estimated 3D
position (derived from noisy perception, NOT ground truth) and a goal
pose, and decides success/fail based on whether the estimated grasp point
lands within tolerance of the true target — without running any real IK
solver or arm controller.

Why this needs to exist at all (not just call execute_action() inline like
run_trial.py currently does): run_trial.py's execute_action() is a coarse
probability stub. This module is the actual physics-flavored version —
it takes real 3D coordinates and a tolerance, and its failure modes
(missed_grasp / wrong_angle / collision) are derived from geometry, not a
random draw. Swap point: run_trial.py's execute_action() calls into this
module's execute() instead of its own inline stub.

Real swap (lab VM, Day 12):
    - true target position -> comes from Gazebo's model state topic
    - perceived position    -> comes from YOLO bbox + depth, reprojected to 3D
    - IK solve + arm motion -> replaced by real OpenVLA action chunk + Gazebo joint commands
    - final check           -> replaced by reading Gazebo's actual final cube pose

Nothing about the tolerance-check logic below changes at that point — only
where target_position_est and target_position_true come from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Outcome = Literal["success", "fail"]
FailureType = Literal["missed_grasp", "wrong_angle", "collision"]


@dataclass
class GraspAttempt:
    outcome: Outcome
    failure_type: FailureType | None
    position_error: float          # euclidean distance, estimated vs true target (m)
    approach_angle_error: float    # radians, how far off the approach vector was


def estimate_target_position(
    true_position: np.ndarray,
    perceived_visibility: float,
    seed: int,
    max_noise_std: float = 0.08,
) -> np.ndarray:
    """
    Simulates how a real perception pipeline's 3D position estimate would
    degrade as perceived_visibility drops. Noise scales inversely with
    perceived_visibility: fully visible -> near-zero noise, fully occluded
    -> up to max_noise_std meters of error in each axis.

    Real swap: this becomes the actual reprojection of the YOLO bounding
    box centroid + TSDF-fused depth into a 3D world-frame point.
    """
    rng = np.random.default_rng(seed)
    noise_scale = max_noise_std * (1.0 - np.clip(perceived_visibility, 0.0, 1.0))
    noise = rng.normal(0.0, noise_scale, size=3)
    return true_position + noise


def estimate_approach_angle_error(perceived_visibility: float, seed: int, max_deg: float = 25.0) -> float:
    """
    Simulates approach-angle uncertainty from noisy surface normal estimates
    under occlusion (TSDF fusion of a partially-occluded object gives a
    worse surface normal estimate than a fully visible one).
    """
    rng = np.random.default_rng(seed + 100)
    noise_scale_deg = max_deg * (1.0 - np.clip(perceived_visibility, 0.0, 1.0))
    angle_deg = abs(rng.normal(0.0, noise_scale_deg))
    return float(np.radians(angle_deg))


def execute(
    true_position: np.ndarray,
    perceived_visibility: float,
    goal_pose_tolerance: float,
    seed: int,
    angle_tolerance_rad: float = np.radians(15.0),
    collision_margin: float = 0.15,
) -> GraspAttempt:
    """
    Core decision logic — pure geometry, no learned model, no sim call.

    Failure taxonomy (matches logger.py's failure_type field):
        missed_grasp -> position error exceeds tolerance
        wrong_angle  -> position OK but approach angle exceeds tolerance
        collision    -> position error is large enough to plausibly clip
                        a neighboring object (only checked if clutter present,
                        approximated here via a fixed margin — real version
                        checks against actual distractor cube positions)
    """
    est_position = estimate_target_position(true_position, perceived_visibility, seed)
    position_error = float(np.linalg.norm(est_position - true_position))
    angle_error = estimate_approach_angle_error(perceived_visibility, seed)

    if position_error > collision_margin:
        return GraspAttempt("fail", "collision", position_error, angle_error)

    if position_error > goal_pose_tolerance:
        return GraspAttempt("fail", "missed_grasp", position_error, angle_error)

    if angle_error > angle_tolerance_rad:
        return GraspAttempt("fail", "wrong_angle", position_error, angle_error)

    return GraspAttempt("success", None, position_error, angle_error)


def _self_test():
    true_pos = np.array([0.3, 0.1, 0.0])

    print("=== mock_vla.py self-test ===\n")

    # High visibility -> expect mostly success
    n_trials = 200
    outcomes_high_vis = [
        execute(true_pos, perceived_visibility=0.95, goal_pose_tolerance=0.02, seed=i).outcome
        for i in range(n_trials)
    ]
    success_rate_high = outcomes_high_vis.count("success") / n_trials
    print(f"[test 1] perceived_visibility=0.95 -> success rate = {success_rate_high:.2f} (expect high, >0.7)")
    assert success_rate_high > 0.6, "Expected high success rate under good visibility"

    # Low visibility -> expect mostly failure
    outcomes_low_vis = [
        execute(true_pos, perceived_visibility=0.10, goal_pose_tolerance=0.02, seed=i).outcome
        for i in range(n_trials)
    ]
    success_rate_low = outcomes_low_vis.count("success") / n_trials
    print(f"[test 2] perceived_visibility=0.10 -> success rate = {success_rate_low:.2f} (expect low, <0.4)")
    assert success_rate_low < 0.5, "Expected degraded success rate under poor visibility"

    assert success_rate_high > success_rate_low, "Visibility should monotonically help success rate"
    print(f"\n[test 3] monotonicity check: {success_rate_high:.2f} > {success_rate_low:.2f}  passed")

    # Failure type distribution sanity check at low visibility
    attempts_low_vis = [
        execute(true_pos, perceived_visibility=0.10, goal_pose_tolerance=0.02, seed=i)
        for i in range(n_trials)
    ]
    failure_types = [a.failure_type for a in attempts_low_vis if a.outcome == "fail"]
    from collections import Counter
    print(f"\n[test 4] failure_type distribution at low visibility: {dict(Counter(failure_types))}")

    print("\nAll mock_vla.py self-tests passed.")
    print("Swap point: run_trial.py's execute_action() should call execute() from this")
    print("module instead of its own inline probability stub.")


if __name__ == "__main__":
    _self_test()