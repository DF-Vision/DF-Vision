"""
scenario_generator.py

Rewrites manipulation_world.sdf per-trial based on experiment grid params,
then (on Linux, with Gazebo installed) launches gz sim and waits for it
to be ready.

TODAY (Windows, no Gazebo): run this in DRY-RUN mode. It will still:
  - generate a valid, unique SDF file per trial
  - let you eyeball / diff the XML rewrites
  - validate the templating logic end-to-end
It just skips the subprocess launch + /clock wait, which need a real
gz sim binary on Linux.

Design notes (publication-rigor pass):
  - occlusion% and clutter_count are now ORTHOGONAL factors. occlusion_panel
    (a dedicated occluder, see manipulation_world.sdf) sits on the active
    camera->target line of sight and is sized by occlusion_pct; distractor
    cubes scatter in a fixed radius band around the target regardless of
    occlusion_pct. Previously both were driven by the same distractor
    placement (tighter scatter = more occlusion), which meant "high
    occlusion + low clutter" and "low occlusion + high clutter" cells
    couldn't exist -- exactly the cells C3's interaction test needs.
  - CAMERA_POSE_RECOVERY is reserved exclusively for C5 and is never used
    as a camera_variant option (that used to collide: camera_variant=1
    pointed at the same pose reserved for C5's second viewpoint, leaving
    C5 nowhere distinct to move the camera to when a gate fired).

Usage:
    python scenario_generator.py --dry-run
    python scenario_generator.py --dry-run --occlusion 60 --clutter 5
"""

import argparse
import math
import random
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_SDF = Path(__file__).parent / "manipulation_world.sdf"
GENERATED_DIR = Path(__file__).parent / "generated_worlds"
GENERATED_DIR.mkdir(exist_ok=True)

# Two genuinely distinct primary-view angles (the camera_variant grid
# factor). CAMERA_POSE_RECOVERY is a separate, third pose -- reserved
# exclusively for C5's runtime fusion move, never assigned here.
CAMERA_POSE_PRIMARY = (0.6, -0.6, 0.8, 0, 0.6, 1.57)
CAMERA_POSE_VARIANT_1 = (0.6, -0.45, 1.05, 0, 1.0, 1.57)  # steeper, more top-down
CAMERA_POSE_RECOVERY = (0.6, 0.6, 0.8, 0, 0.6, -1.57)     # C5 only -- do not add to camera_variant

DISTRACTOR_NAMES = [f"distractor_cube_{i}" for i in range(1, 9)]

# Base positions from manipulation_world.sdf, used to scatter enabled
# distractors near the target, and to place the occlusion panel along
# the camera->target ray.
TARGET_POS = (0.6, 0.0, 0.45)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--occlusion", type=float, default=40.0,
                    help="Target occlusion percent (0-100), approximate")
    p.add_argument("--clutter", type=int, default=3,
                    help="Number of distractor cubes to enable (0-8)")
    p.add_argument("--camera-variant", type=int, choices=[0, 1], default=0)
    p.add_argument("--light-level", type=float, default=1.0,
                    help="Directional light intensity (0.2-2.0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trial-id", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                    help="Generate SDF only, skip gz sim launch (use on Windows)")
    return p.parse_args()


def _set_pose(elem, x, y, z, roll, pitch, yaw):
    pose_elem = elem.find("pose")
    if pose_elem is None:
        pose_elem = ET.SubElement(elem, "pose")
    pose_elem.text = f"{x} {y} {z} {roll} {pitch} {yaw}"


def _set_box_size(model_elem, x, y, z):
    size_elem = model_elem.find(".//box/size")
    if size_elem is not None:
        size_elem.text = f"{x} {y} {z}"


def _set_light_intensity(root, name, intensity):
    for light in root.iter("light"):
        if light.get("name") == name:
            intensity_elem = light.find("intensity")
            if intensity_elem is not None:
                intensity_elem.text = str(intensity)


def scatter_distractors(clutter_count: int, rng: random.Random):
    """
    Returns dict: {model_name: (x, y, z) or None}.
    None means "move off-scene" (parked far below table, out of camera view).

    Placement is a FIXED radius band around the target, independent of
    occlusion_pct -- occlusion is driven entirely by occlusion_panel now
    (see place_occluder below). This decoupling is what lets occlusion%
    and clutter_count vary orthogonally for the C3 interaction analysis,
    instead of clutter objects also being the thing that creates occlusion.
    """
    enabled = DISTRACTOR_NAMES[:clutter_count]
    disabled = DISTRACTOR_NAMES[clutter_count:]

    scatter_radius = 0.22

    placements = {}
    for name in enabled:
        angle = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.08, scatter_radius)
        x = TARGET_POS[0] + r * math.cos(angle)
        y = TARGET_POS[1] + r * math.sin(angle)
        z = TARGET_POS[2] + rng.uniform(-0.01, 0.02)
        placements[name] = (round(x, 4), round(y, 4), round(z, 4))

    for name in disabled:
        placements[name] = None  # parked out of view

    return placements


def place_occluder(cam_pose: tuple, occlusion_pct: float, rng: random.Random):
    """
    Returns (pose_xyz, size_xyz) for occlusion_panel, or (None, None) if
    occlusion_pct is ~0 (parked off-scene entirely so it can't contribute
    any accidental occlusion at the "0%" condition).

    Independent of clutter placement -- positioned along the RAY from the
    trial's actual active camera pose to the target, so it lands on the
    correct line of sight regardless of which camera_variant is active.
    Width scales with occlusion_pct; this is still an approximation (no
    real projection math), same rigor level as the rest of this file --
    actual occlusion is measured afterward via occlusion_estimator.py,
    not assumed from this parameter.
    """
    occ_frac = max(occlusion_pct, 0.0) / 100.0
    if occ_frac <= 0.0:
        return None, None

    cam_xyz = cam_pose[:3]
    frac_along_ray = 0.5  # roughly midway between camera and target
    cx = cam_xyz[0] + frac_along_ray * (TARGET_POS[0] - cam_xyz[0]) + rng.uniform(-0.01, 0.01)
    cy = cam_xyz[1] + frac_along_ray * (TARGET_POS[1] - cam_xyz[1]) + rng.uniform(-0.01, 0.01)
    cz = cam_xyz[2] + frac_along_ray * (TARGET_POS[2] - cam_xyz[2]) + rng.uniform(-0.01, 0.01)

    width = 0.02 + 0.35 * min(occ_frac, 1.0)
    thickness = 0.01
    height = 0.20
    return (round(cx, 4), round(cy, 4), round(cz, 4)), (round(width, 4), thickness, height)


def get_recovery_camera_pose() -> tuple:
    """Fixed second viewpoint reserved exclusively for C5 TSDF fusion
    recovery. Never assigned to a camera_variant at world-generation time
    (see module docstring for the bug this fixes)."""
    return CAMERA_POSE_RECOVERY


def _rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


def move_camera_to_recovery_pose(world_name: str = "manipulation_world", timeout_s: float = 2.0) -> bool:
    """
    Runtime pose-set call for C5: move the already-running rgbd_camera to
    CAMERA_POSE_RECOVERY without regenerating/relaunching the SDF world.

    NOT YET VERIFIED against your installed Gazebo version's exact
    service/message names -- this is a best-effort implementation to
    unblock C5 wiring, not a tested one. Before trusting it in a real
    batch, run:
        gz service -l | grep set_pose
    and confirm the service name and gz.msgs.Pose field layout below
    actually match what your `gz service -l` reports.
    """
    x, y, z, roll, pitch, yaw = CAMERA_POSE_RECOVERY
    qx, qy, qz, qw = _rpy_to_quaternion(roll, pitch, yaw)
    req = (
        f'name: "rgbd_camera", '
        f'position: {{x: {x}, y: {y}, z: {z}}}, '
        f'orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}'
    )
    result = subprocess.run(
        ["gz", "service", "-s", f"/world/{world_name}/set_pose",
         "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
         "--timeout", str(int(timeout_s * 1000)), "--req", req],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def generate_world(args) -> Path:
    tree = ET.parse(BASE_SDF)
    root = tree.getroot()

    rng = random.Random(args.seed + args.trial_id)

    # --- camera pose (two genuinely distinct primary-view variants) ---
    cam_pose = CAMERA_POSE_PRIMARY if args.camera_variant == 0 else CAMERA_POSE_VARIANT_1
    for model in root.iter("model"):
        if model.get("name") == "rgbd_camera":
            _set_pose(model, *cam_pose)

    # --- light intensity ---
    _set_light_intensity(root, "main_light", args.light_level)
    _set_light_intensity(root, "fill_light", args.light_level * 0.5)

    # --- occlusion panel (independent of clutter) ---
    occ_pose, occ_size = place_occluder(cam_pose, args.occlusion, rng)
    for model in root.iter("model"):
        if model.get("name") == "occlusion_panel":
            if occ_pose is None:
                _set_pose(model, 5.0, 5.0, -2.0, 0, 0, 0)
            else:
                x, y, z = occ_pose
                _set_pose(model, x, y, z, 0, 0, 0)
                w, t, h = occ_size
                _set_box_size(model, w, t, h)

    # --- distractor placement (clutter, decoupled from occlusion) ---
    placements = scatter_distractors(args.clutter, rng)
    for model in root.iter("model"):
        name = model.get("name")
        if name in placements:
            pos = placements[name]
            if pos is None:
                # park far below table, out of camera frustum
                _set_pose(model, 5.0, 5.0, -2.0, 0, 0, 0)
            else:
                x, y, z = pos
                _set_pose(model, x, y, z, 0, 0, 0)

    out_path = GENERATED_DIR / f"trial_{args.trial_id:05d}.sdf"
    tree.write(out_path, xml_declaration=True, encoding="unicode")
    return out_path


def launch_gz_sim(world_path: Path, timeout_s: int = 30):
    """Linux-only: launches gz sim headless and waits for /clock topic."""
    proc = subprocess.Popen(
        ["gz", "sim", "-s", "-r", str(world_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    start = time.time()
    while time.time() - start < timeout_s:
        check = subprocess.run(
            ["gz", "topic", "-l"], capture_output=True, text=True
        )
        if "/clock" in check.stdout:
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("gz sim did not become ready within timeout")


def trigger_trial(args):
    """Main entrypoint called by run_trial.py's loop."""
    world_path = generate_world(args)
    print(f"[scenario_generator] wrote {world_path}")

    if args.dry_run:
        print("[scenario_generator] --dry-run set, skipping gz sim launch")
        return None, world_path

    proc = launch_gz_sim(world_path)
    print("[scenario_generator] gz sim ready")
    return proc, world_path


if __name__ == "__main__":
    args = parse_args()
    trigger_trial(args)
