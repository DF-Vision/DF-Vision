"""
run_trial.py -- adds a real-MuJoCo-physics execution path alongside the
existing mock-geometry path, flag-controlled via USE_REAL_MUJOCO_PHYSICS.

WHAT'S NEW IN THIS VERSION
---------------------------------------------------------------------------
USE_REAL_MUJOCO_PHYSICS (module-level flag, default False):
  False (unchanged): build_scenario_poses() -> get_ground_truth() ->
      get_perception() -> mock_vla.execute(), exactly as before. Fast/cheap,
      used for statistical-rehearsal dry runs (Phase 5 dry-run doc,
      mock_vla.py explicitly KEPT for this per earlier decision).
  True (NEW): a real SimpleEnv (action_type='eef_pose', matching real
      trials -- NOT 'delta_joint_angle' like the old Day2/3 scripts) is
      rendered once per trial via mujoco_gap_pipeline.py (NEW MODULE, per
      the Week1 Days1-3 structural decision -- not yet built as of this
      file's writing; this file is written against its intended contract
      below so it slots in with zero further changes to run_trial.py once
      built). execute_grasp.execute_grasp_trial() replaces mock_vla.execute()
      for the actual grasp attempt.

      Perception cadence differs deliberately from the mock path: ONE
      perception snapshot per trial, not N_STEPS_PER_TRIAL=3. execute_grasp.py
      takes a single perceived_target_xyz; real hardware doesn't re-detect
      mid-grasp either. gap_per_step / true_visibility_per_step are still
      populated (as single-element lists) so logger.py's schema stays
      satisfied without special-casing the real-physics path downstream.

      CRITICAL correctness rule carried over from the handoff doc: the
      target position fed into execute_grasp_trial() MUST be the PERCEIVED
      estimate (from mujoco_gap_pipeline's detection+reprojection), NEVER
      env.get_obj_pose()[0] (ground truth) -- feeding ground truth silently
      defeats the entire Gap-driven-failure experimental logic. Enforced
      below by construction: only perceived_target_xyz is ever passed to
      execute_grasp_trial().

  Everything else -- OnlineGateAdapter, TrialRecord construction, session
  bookkeeping, build_experiment_grid() -- is UNCHANGED and shared by both
  paths; only the ground-truth/perception/action-execution block forks.

OBJECT MAPPING (flagged, not yet resolved): real-physics trials currently
reuse MuJoCo's placeholder object names (body_obj_block_1/2) via
mujoco_gap_pipeline's default target/occluder body args -- this project's
5 real study objects (apple/tennis_ball/purse/glass/remote|stapler) do not
yet have corresponding MuJoCo scene assets. OBJECTS[obj_name] below still
drives obj_cfg for the mock path and for gripper/tolerance params on the
real path, but true_visibility/perceived_visibility on the real path come
from whatever bodies are actually in the loaded scene, not from obj_name
directly, until real per-object MuJoCo assets exist. Do not conflate the
two silently -- this is exactly the "remote vs stapler" naming-drift class
of bug this project has hit before.

SCENARIO PLACEMENT + PERCEPTIONFAILURE HANDLING (NEW, 2026-07-17)
---------------------------------------------------------------------------
Two things this file didn't do until now, both needed once
mujoco_gap_pipeline.py was actually built and validated:

1. Real-physics trials now actually POSITION the target/occluder before
   each perception call, via mujoco_day2.force_occlusion_layout() with a
   CAMERA-AWARE base_xy (mujoco_gap_pipeline.pick_camera_aware_base_xy()).
   Previously this file assumed the scene was "already positioned" with no
   mechanism to make that true. base_xy is computed ONCE per environment
   (camera pose is static) and cached in _real_base_xy, not recomputed
   every trial.

   occlusion_pct -> x_shift mapping is a ROUGH, EMPIRICALLY-CALIBRATED
   linear interpolation from the one real sweep run so far (2026-07-17,
   see RUNNING_NOTES.md): x_shift=0.0 -> true_visibility~0.04 (near-total
   occlusion), x_shift>=0.025 -> true_visibility~1.0 (no occlusion). This
   is NOT an exact/validated mapping the way the mock path's geometry is --
   it's a placeholder good enough to get non-degenerate variation across
   the grid's occlusion_pct levels. Flagged explicitly: revisit with a
   denser sweep before treating real-physics occlusion_pct as precisely
   calibrated for any real statistical claim.

2. get_real_perception_and_ground_truth() can now raise
   mujoco_gap_pipeline.PerceptionFailure (e.g. target fully occluded, no
   valid depth in the bbox to reproject a grasp point from). run_trial()
   catches this explicitly and logs a schema-valid TrialRecord using the
   REAL perceived_visibility/true_visibility PerceptionFailure carries
   (see that class's docstring) -- outcome is forced to "fail",
   failure_type="perception_failure", and execute_action_real() is never
   called (there is no perceived_target_xyz to grasp with). This is NOT
   silently dropped or fabricated -- it's a genuine, disclosable failure
   mode distinct from a failed grasp attempt.
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "gazebo"))
sys.path.insert(0, str(Path(__file__).parent / "mock-pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for integration/

from occlusion_estimator import ObjectGeometry, Pose, compute_true_visibility
from gap_computer import compute_gap_features
from conformal_gate import ConformalGate
from logger import TrialRecord, TrialLogger
from integration.prg_client import log_integration_fields
import mock_vla


# --- Policy reproducibility identifier (NEW, 2026-07-17) -----------------
# Per Week 1's "lock the manipulation policy -- checkpoint hash, seed...
# logged in every trial record" requirement. The scripted/IK policy has no
# model weights, but still needs a pinned identifier so a later session
# can reconstruct exactly which grasp logic + IK params produced any given
# trial. Compute ONCE, before any trial collection starts, and treat as
# frozen (same discipline as SCHEMA.md/PREREG.md) -- do not recompute
# mid-collection even if execute_grasp.py changes; that would invalidate
# the whole point of logging it per trial.
import subprocess

def _get_policy_checkpoint_hash() -> str:
    """Git commit hash of the current tree, standing in for a model
    checkpoint hash since the policy is scripted, not learned. Falls back
    to an explicit 'UNKNOWN' string (never silently None) if git isn't
    available, so a missing hash is visible in the log rather than hidden."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent
        ).decode().strip()
    except Exception as e:
        print(f"[WARNING] _get_policy_checkpoint_hash: git rev-parse failed ({e}), "
              f"logging 'UNKNOWN' -- fix before real trial collection starts.")
        return "UNKNOWN"

POLICY_CHECKPOINT_HASH = _get_policy_checkpoint_hash()  # computed once at import time


# --- Real-physics imports are deferred (lazy) inside the USE_REAL_MUJOCO_PHYSICS
# branch itself, not at module top-level -- so the mock/dry-run path never
# requires mujoco/SimpleEnv/execute_grasp to be importable at all. This matters
# because Phase 5-style statistical rehearsals should be runnable in lightweight
# environments without a MuJoCo install.
USE_REAL_MUJOCO_PHYSICS = False  # flip explicitly per run; never silently toggled

# --- Camera intrinsics for the MOCK scenario only (arbitrary but fixed, matches
# occlusion_estimator.py's own self-test scale). Real-physics path gets its own
# intrinsics from mujoco_gap_pipeline's render, not from these constants. ---
CAMERA_INTRINSICS = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
TARGET_DEPTH = 0.6      # meters, target's distance from camera
OCCLUDER_DEPTH = 0.4    # meters, occluder is always closer than target (per depth-ordering rule)

# --- Per-object geometry + detectability (mock path only; real path's actual
# geometry comes from the loaded MuJoCo scene until real object assets exist) ---
OBJECTS = {
    "apple": {"half_extent": 0.04, "noise_level": 0.10},
    "purse": {"half_extent": 0.10, "noise_level": 0.20},
    "glass": {"half_extent": 0.05, "noise_level": 0.30},
}
MUST_HAVE = ["apple", "purse", "glass"]
NICE_TO_HAVE = []

N_STEPS_PER_TRIAL = 3      # mock path only -- real path always uses 1 perception snapshot
SESSION_SIZE = 40
SESSION_BASELINE_TRIALS = 15
GATE_WARMUP_TRIALS = 20

# --- Real-physics x_shift calibration (NEW, 2026-07-17) -- see module
# docstring's "SCENARIO PLACEMENT" section for the rough/placeholder caveat. ---
REAL_MAX_X_SHIFT = 0.025  # meters; x_shift=0.0 -> near-total occlusion,
                           # x_shift>=REAL_MAX_X_SHIFT -> ~no occlusion,
                           # per the 2026-07-17 measured sweep.

# --- Real-physics-only globals: one persistent SimpleEnv instance, built lazily
# on first use so importing this file never requires mujoco to be installed. ---
_real_env = None
_real_base_xy = None  # NEW: camera-aware base_xy, computed once (camera pose is static)
_real_place_xyz = np.array([0.45, 0.0, 0.86], dtype=np.float32)  # placeholder bin-drop point;
                                                                   # TODO: replace with real per-
                                                                   # object grasp target-point once
                                                                   # Day 10's coordinates are finalized

def _get_real_env():
    """Lazily construct the single shared SimpleEnv for real-physics trials.
    action_type='eef_pose' -- matching real trials, NOT 'delta_joint_angle'
    like the old Day2/3 scripts."""
    global _real_env
    if _real_env is None:
        from mujoco_env.y_env import SimpleEnv
        _real_env = SimpleEnv(
            xml_path="asset/scene_so101_y.xml", robot_profile="so101",
            action_type="eef_pose", state_type="joint_angle",
            pick_body_name="body_obj_block_1",   # FIXED 2026-07-17: was body_obj_block_2
                                                   # (the occluder). Must match the
                                                   # target_body passed to
                                                   # mujoco_gap_pipeline.get_perception_and_ground_truth()
                                                   # below, or check_success() judges
                                                   # outcome against a different object
                                                   # than the one Gap was computed for --
                                                   # silently severing the Gap->failure
                                                   # relationship. TODO: still a
                                                   # placeholder body name until real
                                                   # per-object MuJoCo assets exist (see
                                                   # module docstring's OBJECT MAPPING note).
            place_body_name="body_obj_bin",
            spawn_x_range=(0.25, 0.52),
            spawn_y_range=(0.02, 0.22),
            spawn_z_range=(0.815, 0.815),
        )
    return _real_env


# ---------------------------------------------------------------------------
# Online conformal-gate adapter (UNCHANGED -- shared by both paths)
# ---------------------------------------------------------------------------
class OnlineGateAdapter:
    """
    Wraps the real, batch-oriented ConformalGate for live use during trial
    collection. NOT a replacement for Phase 5's proper post-hoc C4 analysis
    (that must re-fit on the frozen dataset's real 70/30 split).
    """

    def __init__(self, alpha: float = 0.10, warmup_trials: int = GATE_WARMUP_TRIALS):
        self.gate = ConformalGate(alpha=alpha)
        self.warmup_trials = warmup_trials
        self._calib_rows: list[dict] = []
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def observe_success_for_calibration(self, gap_initial: float):
        if self._fitted:
            return
        self._calib_rows.append({"Gap_initial": gap_initial})
        if len(self._calib_rows) >= self.warmup_trials:
            calib_df = pd.DataFrame(self._calib_rows)
            self.gate.fit(calib_df)
            self._fitted = True

    def evaluate(self, gap_initial: float) -> dict:
        if not self._fitted:
            return {"gate_fired": False, "nonconformity_score": None, "warmup_phase": True}
        test_df = pd.DataFrame([{"Gap_initial": gap_initial}])
        fired = bool(self.gate.flag(test_df)[0])
        score = float(gap_initial / self.gate._fitted.scale)
        return {"gate_fired": fired, "nonconformity_score": score, "warmup_phase": False}


# ---------------------------------------------------------------------------
# MOCK path (UNCHANGED)
# ---------------------------------------------------------------------------
def build_scenario_poses(occlusion_pct: float, half_extent: float, step: int, rng: np.random.Generator):
    target_geom = ObjectGeometry(width=2 * half_extent, height=2 * half_extent, depth=2 * half_extent)
    target_pose = Pose(x=0.0, y=0.0, z=TARGET_DEPTH)
    occ_half_extent = half_extent * 1.2
    occluder_geom = ObjectGeometry(width=2 * occ_half_extent, height=2 * occ_half_extent, depth=0.02)
    max_offset = 2.0 * (half_extent + occ_half_extent)
    lateral_offset = max_offset * (1.0 - np.clip(occlusion_pct, 0.0, 1.0))
    intrusion = step * 0.02
    lateral_offset = max(0.0, lateral_offset - intrusion)
    lateral_offset += rng.normal(0, 0.01)
    occluder_pose = Pose(x=lateral_offset, y=0.0, z=OCCLUDER_DEPTH)
    return target_pose, target_geom, occluder_pose, occluder_geom


def get_ground_truth(occlusion_pct: float, half_extent: float, step: int, rng: np.random.Generator) -> float:
    target_pose, target_geom, occluder_pose, occluder_geom = build_scenario_poses(
        occlusion_pct, half_extent, step, rng
    )
    return compute_true_visibility(target_pose, target_geom, occluder_pose, occluder_geom,
                                    CAMERA_INTRINSICS)


def get_perception(true_visibility: float, noise_level: float, rng: np.random.Generator) -> float:
    noisy = true_visibility + rng.normal(0, noise_level)
    return float(np.clip(noisy, 0.0, 1.0))


def attempt_fusion_recovery(true_visibility: float, rng: np.random.Generator) -> float:
    return get_perception(true_visibility, noise_level=0.05, rng=rng)


def execute_action_mock(perceived_visibility: float, tolerance: float, seed: int):
    attempt = mock_vla.execute(
        true_position=np.array([0.3, 0.1, 0.0]),
        perceived_visibility=perceived_visibility,
        goal_pose_tolerance=tolerance,
        seed=seed,
    )
    return attempt.outcome, attempt.failure_type


# ---------------------------------------------------------------------------
# REAL-PHYSICS path (NEW, wired 2026-07-17)
# ---------------------------------------------------------------------------
def _get_real_base_xy(env):
    """Lazily compute + cache the camera-aware base_xy (see module docstring).
    Camera pose is static regardless of trial, so this only needs to run once
    per environment, not once per trial."""
    global _real_base_xy
    if _real_base_xy is None:
        import mujoco_gap_pipeline
        _real_base_xy = mujoco_gap_pipeline.pick_camera_aware_base_xy(env)
    return _real_base_xy


def get_real_perception_and_ground_truth(rng: np.random.Generator, occlusion_pct: float):
    """
    Positions the target/occluder for this trial, then takes a single
    perception snapshot from a real MuJoCo render.

    Delegates to mujoco_gap_pipeline.get_perception_and_ground_truth() ->
        (perceived_visibility: float, true_visibility: float,
         perceived_target_xyz: np.ndarray[3])

    perceived_target_xyz must come from bbox-detection + depth-reprojection
    (mirroring what a real detector would give), NEVER from env.get_obj_pose()
    directly -- that would be ground truth and would defeat the experiment.
    Enforced inside mujoco_gap_pipeline.py, not re-checked here.

    Raises
    ------
    mujoco_gap_pipeline.PerceptionFailure if the bbox has no valid depth to
    reproject a grasp point from (e.g. near-total occlusion). Caller
    (run_trial()) must catch this explicitly -- see module docstring.
    """
    import mujoco_gap_pipeline  # deferred import -- only required when this path is used
    from mujoco_day2 import force_occlusion_layout

    env = _get_real_env()
    base_xy = _get_real_base_xy(env)

    # Rough, empirically-calibrated linear mapping -- see module docstring's
    # "SCENARIO PLACEMENT" section. occlusion_pct=0.0 (mock convention: no
    # occlusion) -> x_shift=REAL_MAX_X_SHIFT (occluder far enough away to not
    # overlap). occlusion_pct=1.0 (max occlusion) -> x_shift=0.0 (occluder
    # directly above target, near-total overlap).
    x_shift = REAL_MAX_X_SHIFT * (1.0 - np.clip(occlusion_pct, 0.0, 1.0))
    force_occlusion_layout(env, base_xy=base_xy, x_shift=float(x_shift))

    perceived_visibility, true_visibility, perceived_target_xyz = (
        mujoco_gap_pipeline.get_perception_and_ground_truth(
            env, target_body="body_obj_block_1", occluder_body="body_obj_block_2"
        )
    )
    return perceived_visibility, true_visibility, perceived_target_xyz


def execute_action_real(perceived_target_xyz: np.ndarray):
    """
    Real-physics grasp attempt via execute_grasp.execute_grasp_trial().
    perceived_target_xyz is the ONLY position information passed in -- see
    module docstring's CRITICAL correctness rule.
    """
    from mujoco_env.execute_grasp import execute_grasp_trial
    env = _get_real_env()
    outcome, failure_type, _p_obj_final = execute_grasp_trial(
        env, perceived_target_xyz=perceived_target_xyz, place_xyz=_real_place_xyz
    )
    return outcome, failure_type


# ---------------------------------------------------------------------------
# Session bookkeeping (UNCHANGED -- mock-only; real-physics sessions don't yet
# have a baseline protocol defined, flagged as a gap, not silently reused)
# ---------------------------------------------------------------------------
def run_session_baseline(session_id: int, obj_name: str, seed_base: int) -> float:
    if USE_REAL_MUJOCO_PHYSICS:
        # TODO: real-physics session-baseline protocol not yet defined.
        # Returning 0.0 as an explicit placeholder rather than silently
        # reusing the mock baseline logic, which would misrepresent a
        # real-physics session's actual zero-occlusion failure rate.
        print("[WARNING] run_session_baseline: real-physics baseline not implemented, "
              "returning placeholder 0.0. Do not treat this as a real measurement.")
        return 0.0
    obj_cfg = OBJECTS[obj_name]
    rng = np.random.default_rng(seed_base)
    failures = 0
    for i in range(SESSION_BASELINE_TRIALS):
        true_vis = get_ground_truth(occlusion_pct=0.0, half_extent=obj_cfg["half_extent"], step=0, rng=rng)
        perceived = get_perception(true_vis, obj_cfg["noise_level"], rng)
        outcome, _ = execute_action_mock(perceived, tolerance=0.02, seed=seed_base + i)
        if outcome != "success":
            failures += 1
    return failures / SESSION_BASELINE_TRIALS


# ---------------------------------------------------------------------------
# Core per-trial logic (forks on USE_REAL_MUJOCO_PHYSICS)
# ---------------------------------------------------------------------------
def run_trial(trial_id: int, params: dict, gate_adapter: OnlineGateAdapter,
              logger: TrialLogger) -> TrialRecord:
    seed = params["seed"]
    tolerance = params.get("goal_pose_tolerance", 0.02)
    obj_name = params["object_name"]
    obj_cfg = OBJECTS[obj_name]
    rng = np.random.default_rng(seed)

    perception_failed = False  # NEW: tracked separately from a failed grasp attempt

    if USE_REAL_MUJOCO_PHYSICS:
        # --- Real-physics path: ONE perception snapshot, not N_STEPS_PER_TRIAL ---
        import mujoco_gap_pipeline  # for PerceptionFailure
        try:
            perceived_vis, true_vis, perceived_target_xyz = get_real_perception_and_ground_truth(
                rng, occlusion_pct=params["occlusion_pct"]
            )
        except mujoco_gap_pipeline.PerceptionFailure as e:
            # NEW: genuine failed-perception trial -- log real Gap numbers
            # (carried on the exception, see PerceptionFailure's docstring),
            # but never attempt a grasp with no position to grasp at.
            perception_failed = True
            perceived_target_xyz = None
            perceived_vis = e.perceived_visibility
            true_vis = e.true_visibility
            print(f"[WARNING] trial {trial_id}: PerceptionFailure -- {e} "
                  f"(perceived_vis={perceived_vis}, true_vis={true_vis} still logged)")
            if perceived_vis is None or true_vis is None:
                # Not expected given mujoco_gap_pipeline's pipeline order (see
                # that module's docstring), but if it ever happens, don't
                # fabricate a number -- abort this trial's logging entirely
                # rather than write a schema-valid but meaningless record.
                raise RuntimeError(
                    f"trial {trial_id}: PerceptionFailure with no visibility numbers "
                    f"attached -- cannot log a real TrialRecord, refusing to fabricate one."
                ) from e
        true_vis_per_step = [true_vis]
        perceived_per_step = [perceived_vis]
        timing_per_step = [0.0]  # real-physics timing dominated by execute_grasp_trial(), captured below
    else:
        # --- Mock path: N_STEPS_PER_TRIAL synthetic-geometry steps (unchanged) ---
        true_vis_per_step, perceived_per_step, timing_per_step = [], [], []
        for step in range(N_STEPS_PER_TRIAL):
            t_step = time.perf_counter()
            true_vis_step = get_ground_truth(params["occlusion_pct"], obj_cfg["half_extent"], step, rng)
            perceived_step = get_perception(true_vis_step, obj_cfg["noise_level"], rng)
            true_vis_per_step.append(true_vis_step)
            perceived_per_step.append(perceived_step)
            timing_per_step.append(time.perf_counter() - t_step)

    gap_result = compute_gap_features(perceived_per_step, true_vis_per_step)

    # --- PRG edge-error, same frame/trial_id as the Gap computation above,
    # called before either execute_action_* below acts on the frame. NOT
    # wrapped in a try/except here -- prg_client's internal resolver already
    # catches its own exceptions internally and always returns a valid
    # list[dict] (falling back to a seeded stub on any failure), so a crash
    # in the collaborator's real chain must never propagate up and kill this
    # trial's own Gap logging. No real rgb/depth/masks/camera_intrinsics
    # exist in this pipeline yet (mock path is synthetic-geometry only; the
    # real-physics path doesn't currently surface them out of
    # mujoco_gap_pipeline) -- this always takes the stub path today via
    # prg_config.json's use_stub flag, same as camera_intrinsics=None would
    # force on its own.
    integration_fields = log_integration_fields(
        rgb=None, depth=None, target_name=obj_name, all_object_masks={},
        trial_id=str(trial_id), camera_intrinsics=None,
    )

    true_vis_initial = true_vis_per_step[0]
    perceived_vis_initial = perceived_per_step[0]
    true_vis_final = true_vis_per_step[-1]
    perceived_vis_final = perceived_per_step[-1]

    gate_eval = gate_adapter.evaluate(gap_result.gap_initial)
    fusion_attempted = gate_eval["gate_fired"]

    gap_before_fusion = gap_after_fusion = None

    if USE_REAL_MUJOCO_PHYSICS:
        if perception_failed:
            # NEW: no perceived_target_xyz exists -- do not call
            # execute_action_real() under any circumstance. This is the
            # experimental-logic-preserving equivalent of the CRITICAL
            # ground-truth rule: no position estimate means no grasp, not a
            # ground-truth-position grasp as a fallback.
            outcome, failure_type = "fail", "perception_failure"
        else:
            # Real-physics path does not yet implement fusion recovery (C5) --
            # gate firing is logged, but no re-attempt happens. Flagged, not
            # silently skipped.
            if fusion_attempted:
                print(f"[WARNING] trial {trial_id}: gate fired but real-physics fusion "
                      f"recovery is not implemented -- executing grasp without recovery.")
            t0 = time.perf_counter()
            outcome, failure_type = execute_action_real(perceived_target_xyz)
            timing_per_step[0] = time.perf_counter() - t0
            if outcome == "success" and gate_eval["warmup_phase"]:
                gate_adapter.observe_success_for_calibration(gap_result.gap_initial)
    else:
        if fusion_attempted:
            gap_before_fusion = gap_result.gap_initial
            recovered_perceived = attempt_fusion_recovery(true_vis_final, rng)
            gap_after_fusion = abs(recovered_perceived - true_vis_final)
            if gap_after_fusion < gap_before_fusion:
                outcome, failure_type = execute_action_mock(recovered_perceived, tolerance, seed)
            else:
                outcome, failure_type = "deferred", None
        else:
            outcome, failure_type = execute_action_mock(perceived_vis_final, tolerance, seed)
            if outcome == "success" and gate_eval["warmup_phase"]:
                gate_adapter.observe_success_for_calibration(gap_result.gap_initial)

    failure_flag = 0 if outcome == "success" else 1
    camera_variant = params.get("camera_pose", [0, 0, 0, 0, 0, 0])
    light_level = params.get("light_level", 1.0)
    camera_pose_variant_str = f"cam{camera_variant[2]}_light{light_level}"
    if USE_REAL_MUJOCO_PHYSICS:
        camera_pose_variant_str = "mujoco_real_physics_" + camera_pose_variant_str

    record = TrialRecord(
        trial_id=trial_id,
        session_id=params["session_id"],
        object_id=obj_name,
        session_baseline_failure_rate=params["session_baseline_failure_rate"],
        measured_true_visibility=true_vis_initial,
        true_occlusion=1.0 - true_vis_initial,
        clutter_count=params["clutter_count"],
        perceived_visibility_initial=perceived_vis_initial,
        Gap_initial=gap_result.gap_initial,
        Gap_max=gap_result.gap_max,
        Gap_slope=gap_result.gap_slope,
        failure=failure_flag,
        camera_pose_variant=camera_pose_variant_str,
        policy_checkpoint_hash=POLICY_CHECKPOINT_HASH,
        policy_seed=seed,
        nonconformity_score=gate_eval["nonconformity_score"],
        gate_fired=gate_eval["gate_fired"],
        gate_warmup_phase=gate_eval["warmup_phase"],
        tsdf_fusion_attempted=fusion_attempted,
        Gap_before_fusion=gap_before_fusion,
        Gap_after_fusion=gap_after_fusion,
        outcome=outcome,
        failure_type=failure_type,
        goal_pose_tolerance=tolerance,
        camera_pose=camera_variant,
        light_level=light_level,
        seed=seed,
        gap_per_step=gap_result.gap_trajectory,
        timing_per_step=timing_per_step,
        true_visibility_per_step=true_vis_per_step,
        **integration_fields,
    )
    logger.append_trial(record)
    return record


def build_experiment_grid(n_seeds: int = 1, objects: Optional[list[str]] = None) -> list[dict]:
    objects = objects or list(OBJECTS.keys())
    grid, seed_counter = [], 0
    for obj_name in objects:
        for occlusion_pct in [0.0, 0.2, 0.4, 0.6, 0.8]:
            for clutter_count in [0, 1, 3, 5]:
                for camera_variant in range(2):
                    for light_level in [0.5, 1.0]:
                        for _ in range(n_seeds):
                            grid.append({
                                "object_name": obj_name,
                                "occlusion_pct": occlusion_pct,
                                "clutter_count": clutter_count,
                                "camera_pose": [0, 0, camera_variant, 0, 0, 0],
                                "light_level": light_level,
                                "seed": seed_counter,
                                "goal_pose_tolerance": 0.02,
                            })
                            seed_counter += 1
    return grid


def reset_log_files(log_path: Path):
    freeze_marker = Path(str(log_path) + ".FROZEN")
    if log_path.exists():
        os.remove(log_path)
    if freeze_marker.exists():
        os.remove(freeze_marker)


def main(n_trials: Optional[int] = None):
    log_name = "trials_mujoco_real_physics.jsonl" if USE_REAL_MUJOCO_PHYSICS else "trials_geometry.jsonl"
    log_path = Path(__file__).parent / "mock-pipeline" / "data" / log_name
    reset_log_files(log_path)
    logger = TrialLogger(str(log_path))

    grid = build_experiment_grid(n_seeds=1)
    if n_trials is not None:
        step = max(1, len(grid) // n_trials)
        grid = grid[::step][:n_trials]

    gate_adapter = OnlineGateAdapter(alpha=0.10, warmup_trials=GATE_WARMUP_TRIALS)
    session_cache: dict[int, float] = {}

    print(f"Running {len(grid)} trials (USE_REAL_MUJOCO_PHYSICS={USE_REAL_MUJOCO_PHYSICS}) -> {log_path}")
    for trial_id, params in enumerate(grid):
        session_id = trial_id // SESSION_SIZE
        if session_id not in session_cache:
            session_cache[session_id] = run_session_baseline(
                session_id, params["object_name"], seed_base=100_000 + session_id
            )
        params["session_id"] = session_id
        params["session_baseline_failure_rate"] = session_cache[session_id]

        record = run_trial(trial_id, params, gate_adapter, logger)
        if trial_id % 10 == 0:
            print(f"  trial {trial_id:4d} obj={record.object_id:12s} outcome={record.outcome:9s} "
                  f"gap={record.Gap_initial:.3f} gate_fired={record.gate_fired} "
                  f"warmup={record.gate_warmup_phase}")
    print("Done. Run `python3 run_trial.py smoke` to re-validate end-to-end with 10 trials.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        main(n_trials=10)
    else:
        main(n_trials=30)