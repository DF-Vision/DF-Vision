"""
execute_grasp.py

FIX (this version): the original single-phase "drive straight at target,
then close" approach was found (2026-07-XX MuJoCo grasp smoke test) to
consistently miss the object -- tip converged to within ~1cm of target
(comparable to the object's own 2.5cm size) but object_moved stayed at
~2mm across all 8 trials regardless of occlusion, meaning the gripper
closed on air every time.

Root cause: no hover-then-descend staging (approaches from wherever the
arm happens to be, at whatever orientation it happens to have), and no
top-down orientation lock -- rx,ry,rz were always left at 0 delta, so the
arm never enforced a consistent grasp angle. This mirrors
scripted_fsm_controller.py's PHASES design (above_block -> descend -> grip
-> lift), which already solves this correctly -- ported that staging here
instead of re-deriving it.

Also tightened APPROACH_TOL below object half-extent (was 0.01m vs a
0.0125m half-extent block -- comparable to the object's own size, not a
real tolerance). HOVER_HEIGHT/DESCEND_TOL follow the same "hover well
above, then descend straight down" pattern as the FSM controller.
"""
import numpy as np
import mujoco
import cv2
import os
import mujoco_render
from mujoco_env.transforms import r2rpy

HOVER_HEIGHT = 0.06      # matches scripted_fsm_controller.py's APPROACH_Z_OFFSET
APPROACH_TOL = 0.008     # tighter than before; still > IK jitter, < half object size
DESCEND_TOL = 0.004      # final descent needs to be tighter than hover approach


def _get_grasp_point(env):
    """
    Return the world position of the actual jaw tip ('gripperframe' site),
    not the palm (tcp_body_name/'gripper'). check_success() notes the palm
    sits ~98mm above the jaw tips -- driving the palm straight to the
    perceived target misses the object by that offset. Falls back to the
    palm-minus-0.098 approximation only if the site isn't found, matching
    check_success()'s own fallback.
    """
    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')
    if site_id >= 0:
        return env.data.site_xpos[site_id].copy()
    p_palm = env.get_ee_pose()[:3]
    return p_palm - np.array([0.0, 0.0, 0.098], dtype=np.float32)


def _move_to(env, target_xyz, tol, max_steps, gripper_cmd, lock_xy_only=False):
    """Shared position-servo loop. lock_xy_only=True freezes XY once close
    so the vertical descent doesn't drift laterally (mirrors the FSM
    controller's approach_xy locking during descend)."""
    locked_xy = None
    for _ in range(max_steps):
        p_tip = _get_grasp_point(env)
        delta = target_xyz - p_tip
        if lock_xy_only and locked_xy is not None:
            delta[:2] = locked_xy - p_tip[:2]
        if np.linalg.norm(delta) < tol:
            return p_tip, True
        if lock_xy_only and locked_xy is None and np.linalg.norm(delta[:2]) < tol:
            locked_xy = target_xyz[:2].copy()
        step_delta = np.clip(delta, -0.02, 0.02)
        action = np.concatenate([step_delta, [0, 0, 0], [gripper_cmd]])
        env.step(action)
        env.step_env()
    return _get_grasp_point(env), False


def execute_grasp_trial(env, perceived_target_xyz: np.ndarray,
                         place_xyz: np.ndarray, max_steps: int = 60):
    """
    env: SimpleEnv instance, constructed with action_type='eef_pose'
    (default -- but Day 2/3 smoke-test scripts override to
    'delta_joint_angle', don't copy that constructor call for real trials).
    """
    hover_xyz = perceived_target_xyz + np.array([0.0, 0.0, HOVER_HEIGHT], dtype=np.float32)

    # Phase 1a: hover above target (XY + height), gripper open
    _move_to(env, hover_xyz, APPROACH_TOL, max_steps, gripper_cmd=0.0)


    # ---------- DEBUG IMAGE ----------
    rgb3, _, _, _ = mujoco_render.render_rgbd(env, "sideview")

    os.makedirs("/workspace/grasp_debug", exist_ok=True)

    cv2.imwrite(
        "/workspace/grasp_debug/post_hover.png",
        cv2.cvtColor(rgb3, cv2.COLOR_RGB2BGR),
    )
    print("Saved /workspace/grasp_debug/post_hover.png")


    # Phase 1b: descend straight down onto target, XY locked once aligned
    final_tip, converged = _move_to(
        env, perceived_target_xyz, DESCEND_TOL, max_steps,
        gripper_cmd=0.0, lock_xy_only=True,
    )
    final_delta = perceived_target_xyz - final_tip
    print(f"Descend ended: final error = {np.linalg.norm(final_delta):.4f} m "
            f"(want <{DESCEND_TOL}), converged={converged}")
    print(f"  final tip = {final_tip}")
    print(f"  target    = {perceived_target_xyz}")

    print("\n===== GEOMS =====")
    for i in range(env.model.ngeom):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i)
        print(i, name)
    print("=================\n")

    print("\n===== SITES =====")
    for i in range(env.model.nsite):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_SITE, i)
        print(i, name)
    print("=================\n")

    # # ---------- GEOMETRY DEBUG ----------
    # cube = env.env.get_p_body(env.pick_body_name)
    
    # for name in ["gripperframe"]:
    #     sid = mujoco.mj_name2id(
    #         env.model,
    #         mujoco.mjtObj.mjOBJ_SITE,
    #         name
    #     )
    # print(f"{name}: {env.data.site_xpos[sid]}")

    # for name in ["fixed_jaw_pad", "moving_jaw_pad"]:
    #     gid = mujoco.mj_name2id(
    #         env.model,
    #         mujoco.mjtObj.mjOBJ_GEOM,
    #         name
    #     )
    # print(f"{name}: {env.data.geom_xpos[gid]}")

    # print(f"cube center        : {cube}")

    # if "fixed_jaw_pad" in pts and "moving_jaw_pad" in pts:
    #     jaw_mid = 0.5 * (pts["fixed_jaw_pad"] + pts["moving_jaw_pad"])
    #     print(f"jaw midpoint       : {jaw_mid}")
    #     print(f"jaw error          : {cube - jaw_mid}")

    # if "gripperframe" in pts:
    #     print(f"frame error        : {cube - pts['gripperframe']}")
    # # -----------------------------------

    # ---------- DEBUG IMAGE ----------
    rgb, depth_mm, _, _ = mujoco_render.render_rgbd(env, "sideview")

    os.makedirs("/workspace/grasp_debug", exist_ok=True)

    cv2.imwrite(
        "/workspace/grasp_debug/pre_close.png",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    )



    print("Saved pre-close image to /workspace/grasp_debug/pre_close.png")



    # ---------- DEBUG TOP VIEW ----------
    rgb2, _, _, _ = mujoco_render.render_rgbd(env, "topview")

    cv2.imwrite(
        "/workspace/grasp_debug/pre_close_top.png",
        cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR),
    )

    print("Saved /workspace/grasp_debug/pre_close_top.png")
    # ----------------------------------

    # ---------- DEBUG: gripper joint ----------
    gripper_jid = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "gripper",
    )
    gripper_qpos_addr = env.model.jnt_qposadr[gripper_jid]

    print(
        f"  gripper joint angle BEFORE close: "
        f"{env.data.qpos[gripper_qpos_addr]:.4f} "
        f"(range [-0.175, 1.745])"
    )
    # -----------------------------------------

    # ---------- FRAME DEBUG ----------
    body_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "gripper"
    )

    site_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_SITE,
        "gripperframe"
    )

    fixed_pad_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "fixed_jaw_pad"
    )

    moving_pad_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "moving_jaw_pad"
    )

    print("\n========== PRE-CLOSE GEOMETRY ==========")
    print("Cube center        :", env.get_obj_pose()[0])
    print("Gripper body       :", env.data.xpos[body_id])
    print("Gripperframe site  :", env.data.site_xpos[site_id])
    print("Fixed jaw pad      :", env.data.geom_xpos[fixed_pad_id])
    print("Moving jaw pad     :", env.data.geom_xpos[moving_pad_id])

    pad_mid = (
        env.data.geom_xpos[fixed_pad_id] +
        env.data.geom_xpos[moving_pad_id]
    ) / 2.0

    print("Jaw midpoint       :", pad_mid)
    print("========================================\n")

    body_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "gripper"
    )

    site_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_SITE,
        "gripperframe"
    )

    fixed_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "fixed_jaw_pad"
    )

    moving_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "moving_jaw_pad"
    )

    cube, _ = env.get_obj_pose()

    gripper_body = env.data.xpos[body_id]
    gripper_frame = env.data.site_xpos[site_id]
    fixed_pad = env.data.geom_xpos[fixed_id]
    moving_pad = env.data.geom_xpos[moving_id]

    jaw_mid = (fixed_pad + moving_pad) / 2.0

    print("\n========== PRE-CLOSE ==========")
    print("Cube           :", cube)
    print("Gripper body   :", gripper_body)
    print("Gripper frame  :", gripper_frame)
    print("Fixed jaw      :", fixed_pad)
    print("Moving jaw     :", moving_pad)
    print("Jaw midpoint   :", jaw_mid)
    print("Perceived target :", perceived_target_xyz)

    print("Body error to target  :", np.linalg.norm(gripper_body - perceived_target_xyz))
    print("Frame error to target :", np.linalg.norm(gripper_frame - perceived_target_xyz))
    print("Jaw error to target   :", np.linalg.norm(jaw_mid - perceived_target_xyz))
    print("Frame error    :", np.linalg.norm(gripper_frame - cube))
    print("Jaw error      :", np.linalg.norm(jaw_mid - cube))
    print("===============================\n")

    # ---------- ORIENTATION DEBUG ----------
    p_tcp, R_tcp = env.env.get_pR_body(env.tcp_body_name)

    print("\n========== ORIENTATION ==========")
    print("TCP position:")
    print(p_tcp)

    print("\nTCP rotation matrix:")
    print(R_tcp)

    approach = gripper_frame - gripper_body
    approach = approach / np.linalg.norm(approach)

    print("\nTCP RPY (deg):")
    print(np.rad2deg(r2rpy(R_tcp)))

    print("\nApproach direction (body -> gripperframe):")
    print(approach)


    jaw_axis = moving_pad - fixed_pad
    jaw_axis = jaw_axis / np.linalg.norm(jaw_axis)

    print("\nJaw opening direction (fixed -> moving):")
    print(jaw_axis)

    print("=================================\n")

    # Phase 2: close gripper -- keep closing until the joint is actually near
    # the closed position instead of assuming 10 simulation steps is enough.

    CLOSE_TARGET_RAD = -0.175      # fully closed
    CLOSE_TOL_RAD = 0.15

    for _ in range(150):
        action = np.zeros(7, dtype=np.float32)
        action[-1] = 1.0           # close command

        env.step(action)
        env.step_env()

        current_angle = env.data.qpos[gripper_qpos_addr]

        print(
            f"  EE orientation at grasp: {env.get_ee_pose()[3:]}, "
            f"gripper={current_angle:.4f}"
        )

        if abs(current_angle - CLOSE_TARGET_RAD) < CLOSE_TOL_RAD:
            print(f"  Gripper reached closed position ({current_angle:.4f})")
            break

    print("\n========== CONTACTS AFTER CLOSE ==========")

    for i in range(env.data.ncon):
        con = env.data.contact[i]

        g1_name = mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            con.geom1,
        )

        g2_name = mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            con.geom2,
        )

        print(
            f"{i}: "
            f"geom{con.geom1} ({g1_name}) <--> "
            f"geom{con.geom2} ({g2_name})"
        )

    print("=========================================")

    # Lift

    # ---------- DEBUG: after lift attempt ----------
    rgb_pl, _, _, _ = mujoco_render.render_rgbd(env, "sideview")
    cv2.imwrite("/workspace/grasp_debug/post_lift.png", cv2.cvtColor(rgb_pl, cv2.COLOR_RGB2BGR))
    print("Saved /workspace/grasp_debug/post_lift.png")

    # Phase 3: lift (relative increment -- offset-agnostic, no change needed)
    for _ in range(20):
        action = np.array([0, 0, 0.02, 0, 0, 0, 1.0], dtype=np.float32)
        env.step(action)
        env.step_env()


    # ---------- DEBUG: post-close / post-lift ----------
    rgb_pc, _, _, _ = mujoco_render.render_rgbd(env, "sideview")
    cv2.imwrite("/workspace/grasp_debug/post_close.png", cv2.cvtColor(rgb_pc, cv2.COLOR_RGB2BGR))
    print("Saved /workspace/grasp_debug/post_close.png")

    # Phase 4: move to place location (bin), gripper still closed
    for _ in range(max_steps):
        p_tip = _get_grasp_point(env)
        delta = place_xyz - p_tip
        if np.linalg.norm(delta) < 0.01:
            break
        step_delta = np.clip(delta, -0.02, 0.02)
        action = np.concatenate([step_delta, [0, 0, 0], [1.0]])
        env.step(action)
        env.step_env()

    # Phase 5: open gripper, retract
    for _ in range(10):
        action = np.array([0, 0, 0.01, 0, 0, 0, 0.0], dtype=np.float32)
        env.step(action)
        env.step_env()

    success = env.check_success()
    p_obj, _ = env.get_obj_pose()

    if success:
        return "success", None, p_obj
    return "fail", "missed_grasp", p_obj