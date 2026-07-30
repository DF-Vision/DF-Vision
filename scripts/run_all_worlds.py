#!/usr/bin/env python3
"""
run_all_worlds.py
------------------
Runs the full BPS (Boundary Pose Sensitivity) pipeline — Gazebo world,
depth_publisher.py, depth_to_pointcloud.py, and log_results.py (baseline
+ proposed) — across one or more worlds, back to back, with no manual
per-terminal flag-typing.

Why Python and not bash:
  - We need real process-group lifecycle management (launch in its own
    group, SIGTERM-then-SIGKILL teardown, clean up on Ctrl-C/any error)
    across ~4 long-lived background processes per world. That's `set -m`
    / job-control gymnastics in bash; it's a few lines of subprocess +
    os.killpg in Python.
  - Readiness checks need to parse command output (`ros2 topic list`,
    `ros2 topic echo --once`) and retry with a real deadline loop, not a
    fixed sleep — easy control flow in Python, and this project already
    has a Python precedent for this exact pattern (process_all_rooms.py).
  - The failure model ("fail loudly, stop everything, don't touch the
    next five worlds") wants structured exception handling, not `set -e`
    plus trap juggling.

Usage:
    # Run all six worlds, baseline + proposed, 20 trials each (defaults)
    python3 run_all_worlds.py

    # Run a subset
    python3 run_all_worlds.py --worlds world1_baseline world4_support_scene

    # Override trial count / package / model checkpoint
    python3 run_all_worlds.py --trials 10 --pkg bps_pipeline \\
        --load-from checkpoints/depth_anything_v2_metric_hypersim_vits_NewModified1.pth \\
        --encoder vits --model-type metric --max-depth 20.0

Exit behaviour:
    Any failed step (Gazebo never comes up, a topic never starts
    publishing, log_results.py exits non-zero, etc.) is printed loudly,
    everything launched for that world is torn down, and the script
    exits non-zero immediately — it does NOT continue to the next
    world. A bad run produces at most one world of bad/missing data,
    never six.
"""

import argparse
import os
import signal
import subprocess
import sys
import time

# ── World registry ───────────────────────────────────────────────────────
# Single source of truth mapping world_name -> its SDF file. Adding a
# seventh world means adding one line here, not touching any of the
# per-step logic below.
WORLDS_DIR = "worlds"
SCRIPTS_DIR = "scripts"   
ALL_WORLDS = {
    "world1_baseline":        "world1_baseline.sdf",
    "world2_dense_clutter":   "world2_dense_clutter.sdf",
    "world3_thin_objects":    "world3_thin_objects.sdf",
    "world4_support_scene":   "world4_support_scene.sdf",
    "world5_occlusion_scene": "world5_occlusion_scene.sdf",
    "world6_dense_mixed":     "world6_dense_mixed.sdf",
}

LOG_DIR = os.path.expanduser("~/bps_run_logs")

# ── Bridge topic names ────────────────────────────────────────────────────
# Ground truth: each world SDF's viewpoint-1 camera sensors declare
# <topic>rgb_camera</topic>, <topic>depth_camera</topic>,
# <topic>semantic_camera</topic> (verified directly in the SDF files —
# world*.sdf, sensor name='rgb_camera'/'depth_camera1'/'semantic_camera').
# gz-sim's camera sensor auto-publishes CameraInfo on "<topic>/camera_info"
# alongside the image on "<topic>" — this is standard gz-sensors behavior
# across Ignition/Harmonic, but I have NOT verified it by running this
# specific Gazebo build. Before trusting this end-to-end, run one world
# and `gz topic -l | grep camera_info` to confirm the exact topic name —
# if it differs, update RGB_CAMERA_INFO_TOPIC below (the one place it's
# defined) and nothing else needs to change.
RGB_CAMERA_TOPIC        = "/rgb_camera"
RGB_CAMERA_INFO_TOPIC    = "/camera_info"
DEPTH_CAMERA_TOPIC       = "/depth_camera"
SEMANTIC_CAMERA_TOPIC    = "/semantic_camera/labels_map"


class StepFailed(RuntimeError):
    """Raised for any per-world step failure; carries a human-readable reason."""
    pass


# ── Process helpers ──────────────────────────────────────────────────────
def popen_group(cmd, log_path):
    """
    Launch cmd in its own process group, redirecting stdout/stderr to
    log_path, so it can be torn down as a unit (including anything it
    forks, e.g. gz sim's child processes) without touching this runner
    or sibling processes.
    """
    log_f = open(log_path, "w")
    return subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT, preexec_fn=os.setsid
    )


def kill_process_group(proc, timeout=5):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def wait_for_gz_topic(topic, timeout, poll=0.5):
    """Poll `gz topic -l` until `topic` is registered (Gazemo-side readiness)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            listed = subprocess.run(
                ["gz", "topic", "-l"], capture_output=True, text=True, timeout=5
            ).stdout
        except Exception:
            listed = ""
        if topic in listed:
            return True
        time.sleep(poll)
    return False


def wait_for_ros_topic_publishing(topic, timeout, sample_window=10.0):
    """
    Block until `topic` is actually publishing data — not merely
    registered.

    Previously used `ros2 topic echo --once`, which gambles on catching
    one specific message within a short per-attempt window. Under CPU
    contention (Gazebo + inference running simultaneously on modest
    hardware) this produced repeated false-negative timeouts even when
    the topic was demonstrably publishing fine (confirmed via `gz topic
    -f` and the subscriber's own internal frame-received counters).

    `ros2 topic hz` instead samples continuously over `sample_window`
    seconds and reports a real rate if ANY messages arrive during that
    window — far more forgiving of one slow/missed message, while still
    confirming genuine data flow rather than just topic *existence*.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["ros2", "topic", "hz", "-w", "2", topic],
                capture_output=True, text=True, timeout=sample_window,
            )
            if "average rate" in result.stdout:
                return True
        except subprocess.TimeoutExpired as e:
            # ros2 topic hz blocks until killed; if it printed a rate
            # before the timeout fired, that still counts as success —
            # subprocess.run captures partial output on TimeoutExpired.
            output = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="ignore")
            if "average rate" in output:
                return True
        time.sleep(0.5)
    return False


def run_blocking_step(cmd, log_path, description):
    """Run a foreground step (log_results.py) to completion; raise on failure."""
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise StepFailed(
            f"{description} exited with code {result.returncode} "
            f"(see {log_path})"
        )


def log(msg):
    print(f"[run_all_worlds] {msg}", flush=True)


# ── Per-world pipeline ────────────────────────────────────────────────────
def run_world(world_name, sdf_file, args):
    """
    Run the full pipeline for one world. Raises StepFailed on any error;
    caller is responsible for teardown via the `background` list this
    function populates as it goes, even on failure.
    """
    world_path = os.path.join(WORLDS_DIR, sdf_file)
    world_log_dir = os.path.join(LOG_DIR, world_name)
    os.makedirs(world_log_dir, exist_ok=True)

    background = []  # (name, Popen) pairs, in launch order, for teardown

    def launch_background(name, cmd):
        log(f"  starting {name}: {' '.join(cmd)}")
        proc = popen_group(cmd, os.path.join(world_log_dir, f"{name}.log"))
        background.append((name, proc))
        return proc

    try:
        # 1. Gazebo
        log(f"Launching Gazebo world '{world_name}' ({world_path})")
        gz_proc = launch_background(
            "gazebo",
            ["bash", "-c", f"LIBGL_ALWAYS_SOFTWARE=1 gz sim -s -r {world_path}"],
        )
        if not wait_for_gz_topic("/depth_camera", args.gz_timeout):
            raise StepFailed(
                f"Gazebo did not publish /depth_camera within "
                f"{args.gz_timeout}s for world '{world_name}'"
            )
        if gz_proc.poll() is not None:
            raise StepFailed(f"Gazebo process for '{world_name}' exited early")
        log("  Gazebo ready.")

        # 2. ros_gz_bridge — translates Gazebo's native topics into ROS 2
        # topics. Without this, depth_publisher.py never receives an RGB
        # frame and depth_to_pointcloud.py never receives CameraInfo, no
        # matter how long anything downstream waits.
        log("Starting ros_gz_bridge (parameter_bridge)")
        launch_background(
            "ros_gz_bridge",
            [
                "ros2", "run", "ros_gz_bridge", "parameter_bridge",
                f"{RGB_CAMERA_TOPIC}@sensor_msgs/msg/Image[gz.msgs.Image",
                f"{RGB_CAMERA_INFO_TOPIC}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
                f"{DEPTH_CAMERA_TOPIC}@sensor_msgs/msg/Image[gz.msgs.Image",
                f"{SEMANTIC_CAMERA_TOPIC}@sensor_msgs/msg/Image[gz.msgs.Image",
            ],
        )
        # Reuse wait_for_ros_topic_publishing — same "actual data flow,
        # not just process-started" check used for every other node.
        # Check RGB + CameraInfo specifically: those are the two topics
        # depth_publisher.py and depth_to_pointcloud.py actually consume
        # for this pipeline (see reconciliation notes below); depth and
        # semantic are bridged per your request but aren't on this
        # pipeline's critical path today.
        if not wait_for_ros_topic_publishing(RGB_CAMERA_TOPIC, args.node_timeout):
            raise StepFailed(
                f"{RGB_CAMERA_TOPIC} never started publishing within "
                f"{args.node_timeout}s — ros_gz_bridge likely failed to "
                f"start, or Gazebo isn't actually producing frames, for "
                f"world '{world_name}'"
            )
        if not wait_for_ros_topic_publishing(RGB_CAMERA_INFO_TOPIC, args.node_timeout):
            raise StepFailed(
                f"{RGB_CAMERA_INFO_TOPIC} never started publishing within "
                f"{args.node_timeout}s for world '{world_name}'"
            )
        log(f"  ros_gz_bridge ready ({RGB_CAMERA_TOPIC}, "
            f"{RGB_CAMERA_INFO_TOPIC} publishing).")

        # 3. depth_publisher.py
        # NOTE: rgb_topic default in depth_publisher.py ('/camera/rgb/
        # image_raw') does NOT match the bridged topic name
        # ('/rgb_camera') — passing it explicitly here, rather than
        # relying on the stale default, is what actually connects this
        # node to the bridge's output.
        log("Starting depth_publisher.py")
        launch_background(
            "depth_publisher",
            [
                "python3", os.path.join(SCRIPTS_DIR, "depth_publisher.py"), "--ros-args",
                "-p", f"rgb_topic:={RGB_CAMERA_TOPIC}",
                "-p", f"load_from:={args.load_from}",
                "-p", f"encoder:={args.encoder}",
                "-p", f"model_type:={args.model_type}",
                "-p", f"max_depth:={args.max_depth}",
            ],
        )
        if not wait_for_ros_topic_publishing("/depth_model/depth_image", args.node_timeout):
            raise StepFailed(
                f"/depth_model/depth_image never started publishing within "
                f"{args.node_timeout}s — depth_publisher.py likely failed "
                f"to start for world '{world_name}'"
            )
        log("  depth_publisher.py ready (/depth_model/depth_image publishing).")

        # 4. depth_to_pointcloud.py — world_name is set here, in exactly
        #    one place, so it can never be a forgotten/mistyped flag.
        log(f"Starting depth_to_pointcloud.py (world_name={world_name})")
        launch_background(
            "depth_to_pointcloud",
            [
                "python3", os.path.join(SCRIPTS_DIR, "depth_to_pointcloud.py"), "--ros-args",
                "-p", f"world_name:={world_name}",
                "-p", f"depth_source_kind:={args.depth_source_kind}",
            ],
        )
        if not wait_for_ros_topic_publishing("/depth_model/pointcloud", args.node_timeout):
            raise StepFailed(
                f"/depth_model/pointcloud never started publishing within "
                f"{args.node_timeout}s for world '{world_name}'"
            )
        log("  depth_to_pointcloud.py ready (/depth_model/pointcloud publishing).")

        # 5/6. log_results.py — baseline then proposed. These are
        # foreground/blocking: each runs its trials to completion and
        # exits, so they are NOT added to `background`.
        for condition in ("baseline", "proposed"):
            log(f"Running log_results.py: scene={world_name} condition={condition} "
                f"trials={args.trials}")
            run_blocking_step(
                [
                    "python3", os.path.join(SCRIPTS_DIR, "log_results.py"),
                    "--scene_name", world_name,
                    "--condition", condition,
                    "--trials", str(args.trials),
                    "--results_dir", args.results_dir,
                ],
                os.path.join(world_log_dir, f"log_results_{condition}.log"),
                f"log_results.py ({world_name}/{condition})",
            )
            log(f"  {condition} condition complete.")

        log(f"World '{world_name}' complete.")

    finally:
        # Always tear down everything launched for this world, in
        # reverse order, before returning control (success OR failure) —
        # this is what prevents a scene leaking into the next world.
        log(f"Tearing down background processes for '{world_name}'...")
        for name, proc in reversed(background):
            kill_process_group(proc)
        # Gazebo in particular can leave stragglers; belt-and-suspenders
        # cleanup matching the project's existing convention.
        subprocess.run(["pkill", "-9", "-f", "gz sim"], stderr=subprocess.DEVNULL)
        time.sleep(1)


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worlds", nargs="+", default=list(ALL_WORLDS.keys()),
                         choices=list(ALL_WORLDS.keys()),
                         help="Worlds to run, in order (default: all six).")
    parser.add_argument("--pkg", default="bps_pipeline",
                         help="ROS 2 package name the nodes are run from.")
    parser.add_argument("--trials", type=int, default=20,
                         help="Trials per object per condition (default: 20).")
    parser.add_argument("--results-dir", default=os.path.expanduser("~/grasp_results"),
                         help="Directory log_results.py writes CSVs to.")
    parser.add_argument("--load-from", default="checkpoints/depth_anything_v2_vits.pth",
                         help="Depth model checkpoint path for depth_publisher.py.")
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--model-type", default="relative", choices=["relative", "metric"])
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--gz-timeout", type=float, default=30.0,
                         help="Seconds to wait for Gazebo sensor topics.")
    parser.add_argument("--node-timeout", type=float, default=30.0,
                         help="Seconds to wait for each ROS node's topic to publish.")
    parser.add_argument("--depth-source-kind", default="mde_predicted_z",
                         choices=["gazebo_range_gt", "mde_predicted_z"],
                         help="Passed through to depth_to_pointcloud.py.")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    log(f"Worlds to run: {args.worlds}")
    completed = []
    try:
        for world_name in args.worlds:
            sdf_file = ALL_WORLDS[world_name]
            print("\n" + "=" * 70)
            log(f"WORLD: {world_name}")
            print("=" * 70)
            run_world(world_name, sdf_file, args)
            completed.append(world_name)
    except StepFailed as e:
        log(f"FATAL: {e}")
        log(f"Stopping run. Completed worlds before failure: {completed}")
        sys.exit(1)
    except KeyboardInterrupt:
        log("Interrupted by user. Processes for the current world have been torn down.")
        sys.exit(130)

    print("\n" + "=" * 70)
    log(f"All {len(completed)} world(s) completed successfully: {completed}")
    print("=" * 70)


if __name__ == "__main__":
    main()