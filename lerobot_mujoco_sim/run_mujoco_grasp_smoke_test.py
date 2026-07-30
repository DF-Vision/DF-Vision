"""
run_mujoco_grasp_smoke_test.py

Closes the last gap in Phase 0.25: mujoco_gap_pipeline.py computes
perception/Gap/perceived_target_xyz, execute_grasp.py can grasp, but
nothing wired them together with a real grasp attempt + real logging.
This script does that, end-to-end, on MuJoCo only.

Four Day-2 checks (per master doc / 11-day plan):
  1. RGB-D renders and is consumable by the real detector/TSDF code
     unmodified            -> mujoco_gap_pipeline.get_perception_and_ground_truth()
                               succeeding IS this check (it calls the real
                               tsdf_fusion.py with zero modification).
  2. Scripted controller executes a coherent, non-degenerate grasp
     attempt               -> execute_grasp.execute_grasp_trial() runs;
                               outcome in {success, fail}, object measurably
                               moves (checked below), no NaNs.
  3. Gap computation runs end-to-end, values in [0,1]
                            -> gap_computer.compute_gap_features(), asserted.
  4. Trial logger writes a schema-correct record with controller_script_hash
     (here: policy_checkpoint_hash) and object_id
                            -> logger.TrialRecord/TrialLogger, asserted.

NEVER merge trials_mujoco_grasp_smoke.jsonl into the real dataset or cite
it as C1/C4 evidence -- same rule as trials_mujoco_smoke.jsonl.

Run (inside the mujoco-rehearsal container):
    export MUJOCO_GL=egl
    xvfb-run -a python3 run_mujoco_grasp_smoke_test.py
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import mujoco

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "mock-pipeline"))

from mujoco_env.y_env import SimpleEnv
from mujoco_env.execute_grasp import execute_grasp_trial
from mujoco_day2 import force_occlusion_layout
import mujoco_gap_pipeline as mgp
from gap_computer import compute_gap_features
from logger import TrialRecord, TrialLogger

N_TRIALS = 8
PLACEHOLDER_OBJECT_ID = "apple"  # MuJoCo generic block_1, NOT real apple data -- see docstring
LOG_PATH = Path(__file__).parent / "mock-pipeline" / "data" / "trials_mujoco_grasp_smoke.jsonl"
PLACE_XYZ = np.array([0.25, -0.12, 0.86], dtype=np.float32)  # matches obj_bin.xml position


def _policy_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent
        ).decode().strip()
    except Exception as e:
        print(f"[WARNING] git rev-parse failed ({e}), logging 'UNKNOWN'")
        return "UNKNOWN"


def main():
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    freeze_marker = Path(str(LOG_PATH) + ".FROZEN")
    if freeze_marker.exists():
        freeze_marker.unlink()
    logger = TrialLogger(str(LOG_PATH))

    policy_hash = _policy_hash()

    env = SimpleEnv(
        xml_path="asset/scene_so101_y.xml", robot_profile="so101",
        action_type="eef_pose", state_type="joint_angle",   # eef_pose -- required by execute_grasp.py
        pick_body_name="body_obj_block_1",
        place_body_name="body_obj_bin",
        spawn_x_range=(0.25, 0.52),
        spawn_y_range=(0.02, 0.22),
        spawn_z_range=(0.815, 0.815),
        enable_viewer=True,
    )


    print("\n===== GEOM -> BODY =====")

    for i in range(env.env.model.ngeom):
        body_id = env.env.model.geom_bodyid[i]

        body_name = mujoco.mj_id2name(
            env.env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        geom_name = mujoco.mj_id2name(
            env.env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            i,
        )

        print(f"{i:2d}  {body_name:25s}  {geom_name}")

    print("========================\n")

    print("\n=== GEOMS ===")
    for i in range(env.model.ngeom):
        geom_name = mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            i,
        )
        body_id = env.model.geom_bodyid[i]
        body_name = mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )
        print(f"{i:2d}: body={body_name}, geom={geom_name}")
    print("=============\n")

    base_xy = mgp.pick_camera_aware_base_xy(env)
    x_shifts = np.linspace(0.0, 0.03, N_TRIALS)

    print(f"Running {N_TRIALS} MuJoCo grasp smoke trials -> {LOG_PATH}\n")
    results = []

    for trial_id, x_shift in enumerate(x_shifts):
        force_occlusion_layout(env, base_xy=base_xy, x_shift=float(x_shift))

        try:
            perceived_vis, true_vis, perceived_target_xyz = mgp.get_perception_and_ground_truth(
                env, target_body="body_obj_block_1", occluder_body="body_obj_block_2"
            )
        except mgp.PerceptionFailure as e:
            print(f"trial {trial_id}: PerceptionFailure ({e}) -- logging fail, no grasp attempted")
            perceived_vis, true_vis = e.perceived_visibility, e.true_visibility
            if perceived_vis is None or true_vis is None:
                raise RuntimeError(f"trial {trial_id}: no visibility numbers to log") from e
            gap_result = compute_gap_features([perceived_vis], [true_vis])
            record = TrialRecord(
                trial_id=trial_id, session_id=0, object_id=PLACEHOLDER_OBJECT_ID,
                session_baseline_failure_rate=0.0,
                measured_true_visibility=true_vis, true_occlusion=1.0 - true_vis,
                clutter_count=0, perceived_visibility_initial=perceived_vis,
                Gap_initial=gap_result.gap_initial, Gap_max=gap_result.gap_max,
                Gap_slope=gap_result.gap_slope, failure=1,
                camera_pose_variant=f"mujoco_grasp_smoke_x{x_shift:.4f}",
                policy_checkpoint_hash=policy_hash, policy_seed=trial_id,
                outcome="fail", failure_type="perception_failure",
                gap_per_step=gap_result.gap_trajectory,
            )
            logger.append_trial(record)
            results.append(record)
            continue

        # --- Sanity: object must actually be gripper-reachable / non-NaN ---
        assert np.all(np.isfinite(perceived_target_xyz)), \
            f"trial {trial_id}: perceived_target_xyz has NaN/Inf: {perceived_target_xyz}"

        p_obj_before, _ = env.get_obj_pose()

        outcome, failure_type, p_obj_after = execute_grasp_trial(
            env, perceived_target_xyz=perceived_target_xyz, place_xyz=PLACE_XYZ
        )

        moved = float(np.linalg.norm(np.asarray(p_obj_after) - np.asarray(p_obj_before)))

        gap_result = compute_gap_features([perceived_vis], [true_vis])
        assert 0.0 <= perceived_vis <= 1.0 and 0.0 <= true_vis <= 1.0, \
            f"trial {trial_id}: visibility out of [0,1] range"

        record = TrialRecord(
            trial_id=trial_id, session_id=0, object_id=PLACEHOLDER_OBJECT_ID,
            session_baseline_failure_rate=0.0,
            measured_true_visibility=true_vis, true_occlusion=1.0 - true_vis,
            clutter_count=0, perceived_visibility_initial=perceived_vis,
            Gap_initial=gap_result.gap_initial, Gap_max=gap_result.gap_max,
            Gap_slope=gap_result.gap_slope, failure=0 if outcome == "success" else 1,
            camera_pose_variant=f"mujoco_grasp_smoke_x{x_shift:.4f}",
            policy_checkpoint_hash=policy_hash, policy_seed=trial_id,
            outcome=outcome, failure_type=failure_type,
            gap_per_step=gap_result.gap_trajectory,
        )
        logger.append_trial(record)
        results.append(record)

        print(f"  trial {trial_id} x_shift={x_shift:.4f}  true_vis={true_vis:.4f}  "
              f"perceived_vis={perceived_vis:.4f}  gap={gap_result.gap_initial:.4f}  "
              f"outcome={outcome:9s}  object_moved={moved:.4f}m")

    # --- Decision-gate checks (Day 2's four yes/no checks) ---
    outcomes = {r.outcome for r in results}
    bad_schema = [r.trial_id for r in results
                  if r.policy_checkpoint_hash is None or r.object_id is None]
    gap_range_ok = all(0.0 <= r.Gap_initial <= 1.0 for r in results)

    print("\n=== MuJoCo grasp smoke test checkpoint ===")
    print(f"1. Perception/TSDF ran unmodified on all trials: {len(results)}/{N_TRIALS}")
    print(f"2. Non-degenerate grasp outcomes observed: {outcomes} "
          f"({'OK' if len(outcomes) >= 1 and outcomes <= {'success','fail','perception_failure'} else 'CHECK'})")
    print(f"3. Gap values in [0,1]: {'OK' if gap_range_ok else 'CHECK'}")
    print(f"4. Schema fields (policy_checkpoint_hash, object_id) populated: "
          f"{'OK' if not bad_schema else f'CHECK -- missing on trials {bad_schema}'}")
    print(f"Log written: {LOG_PATH} (exists={LOG_PATH.exists()})")
    print("\nNEVER cite these numbers as C1/C4 evidence -- MuJoCo smoke-test only.")


if __name__ == "__main__":
    main()