#!/usr/bin/env python3
"""
log_results.py
--------------
Runs the full grasp estimation experiment for N trials across all test
objects discovered for the given scene, records every decision, and
outputs a CSV file ready for your paper's results table.

Automatically runs baseline and proposed conditions in sequence.
You swap the model by passing --condition flag (or the `condition` param).

Object list is discovered automatically from
`ground_truth/<scene_name>_pose.json` (produced by export_pose.py) — no
object names, world names, or expected grasp decisions are hard-coded, so
this works unchanged across world1_primitives, world2_household,
world3_kitchen_objects, world4_office_objects, world5_mixed_clutter, and
world6_occlusion.

Usage:
    # Run baseline condition
    ros2 run <pkg> log_results.py --ros-args \
        -p scene_name:=world3_kitchen_objects -p condition:=baseline -p trials:=20

    # Run proposed condition
    ros2 run <pkg> log_results.py --ros-args \
        -p scene_name:=world3_kitchen_objects -p condition:=proposed -p trials:=20

    # Analyse and compare both
    python3 log_results.py --analyse --results_dir ~/grasp_results
"""

import csv
import os
import sys
import json
import glob
import time
import argparse
import collections
from datetime import datetime

import rclpy
from rclpy.node import Node

from grasp_estimator import GraspEstimator


# ── Configuration ────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/grasp_results")
GROUND_TRUTH_DIR    = "ground_truth"

RESULTS_FILE_TEMPLATE = "{output_dir}/results_{condition}_{timestamp}.csv"
SUMMARY_FILE          = "{output_dir}/summary_comparison.csv"

CSV_HEADERS = [
    "trial", "condition", "object_name",
    "estimated_width_m", "estimated_height_m",
    "gt_width_m", "gt_height_m",
    "width_error_m", "height_error_m",
    "decision", "gt_decision", "correct",
    "timestamp"
]

# ── New pilot-experiment schema (DA-V2 baseline vs BP-MDE proposed) ──────
# NEW: written to a SEPARATE file ("trials_<condition>_<timestamp>.csv",
# not "results_<condition>_<timestamp>.csv") so it does NOT collide with
# CSV_HEADERS/RESULTS_FILE_TEMPLATE above. analyse_results() below globs
# "results_*.csv" and reads several old-schema-only columns
# (width_error_m, height_error_m, correct) — it is the one other place
# in this file that consumes this CSV, and it is left completely
# untouched so it keeps working on old-format runs. No other script in
# this codebase parses results_*.csv (checked: evaluate_all.py reads/
# writes its own unrelated baseline_results.csv from .npy depth maps,
# not this CSV at all).
RESULTS_FILE_TEMPLATE_V2 = "{output_dir}/trials_{condition}_{timestamp}.csv"

CSV_HEADERS_V2 = [
    "trial_id", "timestamp", "depth_backbone",
    "object_a", "object_b",
    "true_relation", "predicted_relation", "relation_correct",
    "boundary_sharpness", "minimum_boundary_distance", "normal_consistency",
    "overlap_ratio", "relative_height",
    "grasp_x", "grasp_y", "grasp_z",
    "grasp_success", "task_success", "failure_type",
    "camera_frame_id", "rgb_path", "depth_path", "mask_path",
    "inference_time_ms", "feature_time_ms", "classifier_time_ms",
    # Optional — always None until a camera-to-base calibration mechanism
    # exists; column is present now so no future schema migration is
    # needed once it does.
    "camera_to_base_transform_id",
]

# Fixed failure vocabulary — failure_type is always one of these or ""
# (empty = no failure), never a freeform string.
FAILURE_CODES = (
    "NO_RGB", "NO_DEPTH", "EMPTY_MASK", "EMPTY_POINTCLOUD",
    "INVALID_FEATURES", "RELATION_FAILURE", "NO_VALID_GRASP",
    "TRANSFORM_FAILURE", "ARM_FAILURE",
)

# NEW: log_results.py only ever receives "baseline"/"proposed" via the
# `condition` parameter — it has no visibility into which checkpoint
# depth_publisher.py was actually launched with in another terminal/
# process. This mapping is therefore a NAMING CONVENTION, not a
# verified fact: confirm "baseline"->DA-V2 and "proposed"->BP-MDE match
# how you're actually launching depth_publisher.py for each run.
CONDITION_TO_BACKBONE = {
    "baseline": "DA-V2",
    "proposed": "BP-MDE",
}


def discover_objects(scene_name: str):
    """
    Discover test objects for this scene from
    ground_truth/<scene_name>_pose.json (produced by export_pose.py).

    No object names, world names, or expected decisions are hard-coded —
    the object list is whatever export_pose.py recorded for this scene.
    """
    pose_path = os.path.join(GROUND_TRUTH_DIR, f"{scene_name}_pose.json")
    if not os.path.isfile(pose_path):
        raise FileNotFoundError(
            f"Ground-truth pose file not found: {pose_path}. "
            f"Run export_pose.py for scene '{scene_name}' first."
        )

    with open(pose_path) as f:
        pose_data = json.load(f)

    # Current export_pose.py format: {"scene": "...", "objects": [{"name": "...", ...}, ...]}
    if isinstance(pose_data, dict) and "objects" in pose_data:
        objects = pose_data["objects"]
        object_names = [
            obj.get("name", obj.get("object_name"))
            for obj in objects
        ]
        object_names = [n for n in object_names if n]
    # Backwards compatibility: dict keyed directly by object name
    elif isinstance(pose_data, dict):
        object_names = list(pose_data.keys())
    # Backwards compatibility: flat list of per-object dicts
    elif isinstance(pose_data, list):
        object_names = [
            obj.get("object_name", obj.get("name"))
            for obj in pose_data
        ]
        object_names = [n for n in object_names if n]
    else:
        raise ValueError(
            f"Unrecognised ground-truth pose format in {pose_path}"
        )

    if not object_names:
        raise ValueError(f"No objects found in {pose_path}")

    return object_names


class ResultsLogger:
    def __init__(self, node: Node, scene_name: str, condition: str,
                 n_trials: int, output_dir: str, require_fresh_cloud: bool = True):
        self.node       = node
        self.scene_name = scene_name
        self.condition  = condition
        self.n_trials   = n_trials
        self.output_dir = output_dir
        self.require_fresh_cloud = require_fresh_cloud

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = RESULTS_FILE_TEMPLATE.format(
            output_dir=self.output_dir,
            condition=condition,
            timestamp=timestamp
        )
        self.rows = []

        # NEW: separate CSV/state for the new pilot-experiment schema.
        # Written alongside (not instead of) the old CSV — see the note
        # above CSV_HEADERS_V2.
        self.csv_path_v2 = RESULTS_FILE_TEMPLATE_V2.format(
            output_dir=self.output_dir,
            condition=condition,
            timestamp=timestamp
        )
        self.rows_v2 = []
        self.depth_backbone = CONDITION_TO_BACKBONE.get(condition, condition)
        self._trial_seq = 0  # monotonically increasing across all objects/trials in this run

        self.test_objects = discover_objects(scene_name)

    def run(self):
        """Run all trials for all discovered objects and log results."""
        log = self.node.get_logger()

        total_correct    = 0
        total_trials     = 0
        evaluated_trials = 0

        # GraspEstimator is a full ROS 2 Node — construct it once via its
        # real __init__ (subscriptions, params, ground-truth loading, etc.)
        # and reuse the same instance for every object/trial.
        estimator = GraspEstimator()

        # MOD: GraspEstimator.__init__ reads world_name from its own ROS
        # parameter (default "") and loads ground_truth/<world_name>_pose.json
        # + boundary/semantic mask dirs at construction time. log_results.py
        # never passes world_name via --ros-args, so it stays "" and all of
        # that silently fails to resolve. Set it from scene_name here and
        # manually re-run the world_name-dependent setup that __init__
        # already ran (with the empty value).
        estimator.world_name = self.scene_name
        # MOD: same issue as world_name — roi_x/y/z_range are ROS parameters
        # on GraspEstimator with defaults tuned for a forward-facing robot
        # camera (x=forward,y=left-right,z=up), but this pipeline uses a
        # top-down camera producing points in optical frame (x=right,
        # y=down, z=depth/forward). log_results.py never overrides these,
        # so set them directly. TEMP: wide-open bounds first, to confirm
        # points appear at all before narrowing to the real table region.
        
        estimator._ground_truth_data = estimator._load_ground_truth_json()
        estimator._ground_truth_load_warned = False
        estimator._boundary_masks_path = os.path.join(
            estimator.boundary_masks_dir, estimator.world_name)
        estimator._semantic_masks_path = os.path.join(
            estimator.semantic_masks_dir, estimator.world_name)
        estimator._log_mask_dir_status("boundary masks", estimator._boundary_masks_path)
        estimator._log_mask_dir_status("semantic masks", estimator._semantic_masks_path)

        # MOD: DDS discovery for a freshly-created subscription can take a
        # few seconds to match with depth_to_pointcloud's existing
        # publisher. Under BEST_EFFORT QoS (no retained history), any cloud
        # published before the match completes is dropped, not queued —
        # so the first object tested can burn through its 10s warmup
        # timeout before a single message arrives, even though later
        # objects (same subscription, already matched) succeed in ~1-3s.
        # Spin here, before the per-object loop starts, so discovery
        # latency is absorbed once rather than risking the first object's
        # trial being silently skipped.
        log.info("Warming up pointcloud subscription (DDS discovery)...")
        warmup_deadline = estimator.get_clock().now().nanoseconds + int(25.0 * 1e9)
        while estimator.latest_cloud is None:
            rclpy.spin_once(estimator, timeout_sec=0.1)
            if estimator.get_clock().now().nanoseconds > warmup_deadline:
                log.warning("Pointcloud subscription did not warm up within 25s.")
                break

        for obj_name in self.test_objects:

            log.info("=" * 50)
            log.info(f"Testing object: {obj_name}")
            log.info("=" * 50)

            estimator.object_name  = obj_name

            # MOD: the ROI was a single static box covering the whole
            # table (set once before this loop), so every object's
            # "estimate" was really measuring the same undifferentiated
            # blob — same estimated width for cube/cylinder/sphere/cone.
            # Center a tight box on this object's known ground-truth (x,y)
            # instead. Cloud is in optical frame (x=right, y=down,
            # z=depth), and objects sit on the table at world-frame-ish
            # x/y from the JSON with real depth (z) in the 0.72-0.85m
            # band observed earlier (table surface), well short of the
            # ~1.16m floor sliver — so a modest half-width plus a tight
            # z band isolates the object and excludes the rest of the
            # table/floor.

            gt_entry = estimator._ground_truth_data.get(obj_name) if estimator._ground_truth_data else None
            if gt_entry is not None:
                # MOD: attempting to derive an exact world->cloud coordinate
                # transform to center a tight per-object ROI proved unreliable
                # (best-fit affine solve gave scale=0.73, not the expected 1.0
                # for a rigid transform, indicating an unresolved ambiguity).
                # The depth-margin filter + tight cluster_radius in
                # grasp_estimator.py already isolates individual raised
                # objects correctly on its own (confirmed: cluster_points
                # dropped from 17902 to 2201, matching a single object's
                # scale) — so leave the ROI wide and let that mechanism do
                # the isolation instead of a possibly-wrong coordinate
                # transform.
                estimator.roi_x_range = (-1.0, 1.0)
                estimator.roi_y_range = (-1.0, 1.0)
                estimator.roi_z_range = (0.0, 2.0)
            
            # gt_entry = estimator._ground_truth_data.get(obj_name) if estimator._ground_truth_data else None
            # if gt_entry is not None:
            #     gt_x = float(gt_entry["x"])
            #     gt_y = float(gt_entry["y"])
            #     half_width = 0.15  # metres, generous margin around object footprint
            #     estimator.roi_x_range = (gt_x - half_width, gt_x + half_width)
            #     estimator.roi_y_range = (gt_y - half_width, gt_y + half_width)
            #     estimator.roi_z_range = (0.0, 2.0)  # table surface band, now in world frame
            #     log.info(
            #         f"ROI centered on '{obj_name}' GT pose: "
            #         f"x[{estimator.roi_x_range[0]:.2f},{estimator.roi_x_range[1]:.2f}] "
            #         f"y[{estimator.roi_y_range[0]:.2f},{estimator.roi_y_range[1]:.2f}]"
            #     )
            else:
                log.warning(
                    f"No ground-truth pose found for '{obj_name}' — "
                    f"leaving ROI unchanged (results for this object "
                    f"will likely be unreliable)."
                )

            if self.require_fresh_cloud:
                estimator.latest_cloud = None

            # Warm up — wait for first cloud
            log.info("Waiting for pointcloud...")
            deadline = estimator.get_clock().now().nanoseconds + int(20.0 * 1e9)
            while estimator.latest_cloud is None:
                rclpy.spin_once(estimator, timeout_sec=0.1)
                if estimator.get_clock().now().nanoseconds > deadline:
                    log.error("Timeout waiting for pointcloud. "
                              "Is depth_publisher.py running?")
                    break

            # Run N trials for this object
            for trial in range(1, self.n_trials + 1):
                # MOD: spinning for a fixed 0.5s and then calling estimate()
                # regardless meant most trials fired before depth_publisher.py
                # (which on CPU only produces a new frame every ~3-10s, per
                # `ros2 topic hz /depth_model/pointcloud`) had actually
                # published anything new — estimate() would then see a stale
                # frame and return None, and the trial was logged as
                # "object not detected" even though nothing was wrong with
                # detection itself. Instead, explicitly wait until a new
                # depth frame has actually arrived (via _frames_received,
                # incremented in _depth_callback) before calling estimate(),
                # bounded by a generous timeout so a genuinely stalled
                # depth_publisher still surfaces as a real failure rather
                # than hanging forever.
                frames_before = estimator._frames_received
                frame_deadline = estimator.get_clock().now().nanoseconds + int(15.0 * 1e9)
                while estimator._frames_received <= frames_before:
                    rclpy.spin_once(estimator, timeout_sec=0.5)
                    if estimator.get_clock().now().nanoseconds > frame_deadline:
                        log.warning(
                            f"Trial {trial}/{self.n_trials}: no new depth "
                            f"frame arrived within 15s — proceeding with "
                            f"whatever is latest."
                        )
                        break

                self._trial_seq += 1
                trial_id = f"{self.scene_name}_{self.condition}_{obj_name}_{trial:03d}"

                # NEW: base v2 row — every field defaults to None/blank so a
                # row is always written even if we bail out early below.
                # This is what satisfies "never lose a row just because
                # something failed partway through": we build this dict
                # first, keep filling it in as far as we get, and there is
                # exactly one exit from this trial (bottom of the try
                # block) where it gets appended — no early `return`/
                # `continue` skips it.
                row_v2 = {h: None for h in CSV_HEADERS_V2}
                row_v2["trial_id"]       = trial_id
                row_v2["timestamp"]      = datetime.now().isoformat()
                row_v2["depth_backbone"] = self.depth_backbone
                row_v2["object_a"]       = obj_name
                # object_b / true_relation / predicted_relation /
                # relation_correct: no source. This pipeline's
                # GraspEstimator evaluates ONE object's grasp feasibility;
                # it does not compare a pair of objects' spatial relation.
                # There is no relation-classifier component anywhere in
                # this codebase to populate these from. Left as None.
                # boundary_sharpness / minimum_boundary_distance /
                # normal_consistency / overlap_ratio / relative_height: no
                # source. No feature-extraction module computing these
                # exists yet. Left as None.
                # grasp_x/y/z, grasp_success, task_success: no source.
                # GraspEstimator estimates bounding-box feasibility
                # (can_grasp bool), not a grasp point, and nothing in this
                # pipeline executes a grasp on an arm or reports task
                # outcome. Left as None.
                # rgb_path, depth_path: no source. No per-trial frame
                # saving occurs during log_results.py runs (save_rgb_gz.py
                # / save_depth_gz.py exist but aren't invoked from here).
                # camera_to_base_transform_id: no source (no calibration
                # mechanism exists yet) — left None per your note that
                # it's optional until one does.
                # feature_time_ms, classifier_time_ms: no source — no
                # separate feature-extraction/classifier stage exists to
                # time independently of estimate() below.

                # NEW: real wall-clock timing around the estimation call.
                # NOTE: this times GraspEstimator.estimate() (pointcloud
                # ROI/clustering/dimension-fit), which runs in THIS
                # process. It does NOT time the depth model's own forward
                # pass — that happens inside depth_publisher.py, a
                # separate node/process with no per-frame timing exposed
                # today. Don't read inference_time_ms as depth-model
                # latency; it's grasp-estimation-step latency.
                cloud_was_present_before_call = estimator.latest_cloud is not None
                t0 = time.monotonic()
                try:
                    result = estimator.estimate()
                except Exception as e:
                    # Catch-all for anything unexpected inside estimate()
                    # itself (distinct from the internal `except Exception`
                    # around dimension-fitting, which already returns None).
                    # INVALID_FEATURES is the closest fit in the fixed
                    # vocabulary for "perception-stage computation blew up"
                    # — there is no generic/UNKNOWN code by design (see
                    # requirement 4), so unanticipated failures still need
                    # to land in one of the nine buckets.
                    log.error(f"Trial {trial}/{self.n_trials}: estimate() raised: {e}")
                    row_v2["inference_time_ms"] = round((time.monotonic() - t0) * 1000, 2)
                    row_v2["failure_type"] = "INVALID_FEATURES"
                    self.rows_v2.append(row_v2)
                    total_trials += 1
                    time.sleep(0.5)
                    continue
                row_v2["inference_time_ms"] = round((time.monotonic() - t0) * 1000, 2)

                # camera_frame_id / mask_path: real values from the
                # pipeline's own state, available regardless of whether
                # this trial ultimately succeeds.
                if estimator.latest_cloud is not None:
                    row_v2["camera_frame_id"] = getattr(estimator, "latest_cloud_frame_id", "") or ""
                row_v2["mask_path"] = os.path.join(
                    estimator._semantic_masks_path, f"{obj_name}_mask.npy"
                )

                if result is not None:
                    result = result.to_dict()

                if result is None or not result.get("object_detected", False):
                    # Distinguish the two observable failure modes from
                    # here: no pointcloud at all vs. a pointcloud existed
                    # but the object couldn't be isolated in it (missing
                    # mask file, too few points after filtering, or a
                    # shape mismatch/exception inside dimension-fitting —
                    # GraspEstimator collapses those last three into the
                    # same object_detected=False/None it always has, so
                    # log_results.py can't tell them apart further without
                    # a grasp_estimator.py change, which is out of scope
                    # here).
                    if not cloud_was_present_before_call:
                        row_v2["failure_type"] = "EMPTY_POINTCLOUD"
                    elif result is None:
                        row_v2["failure_type"] = "INVALID_FEATURES"
                    else:
                        row_v2["failure_type"] = "EMPTY_MASK"
                    self.rows_v2.append(row_v2)
                    total_trials += 1
                    log.warning(
                        f"Trial {trial}/{self.n_trials}: object not detected — "
                        f"logged as failure_type={row_v2['failure_type']}"
                    )
                    time.sleep(0.5)
                    continue

                decision = "GRASP" if result["can_grasp"]    else "SKIP"
                gt_dec   = ("GRASP" if result["gt_can_grasp"] else "SKIP") \
                           if result["gt_can_grasp"] is not None else "N/A"
                correct  = result["decision_correct"]

                # Grasp target point (camera-optical frame — see
                # grasp_estimator.py's EstimationResult docstring). Only
                # meaningful when can_grasp is True; logged either way so
                # the CSV shows what was actually computed per trial.
                row_v2["grasp_x"] = result.get("grasp_x")
                row_v2["grasp_y"] = result.get("grasp_y")
                row_v2["grasp_z"] = result.get("grasp_z")

                row = {
                    "trial":              trial,
                    "condition":          self.condition,
                    "object_name":        obj_name,
                    "estimated_width_m":  round(result["estimated_width"],  4),
                    "estimated_height_m": round(result["estimated_height"], 4),
                    "gt_width_m": (
                        round(result["gt_width"], 4)
                        if result["gt_width"] is not None
                        else None
                    ),
                    "gt_height_m": (
                        round(result["gt_height"], 4)
                        if result["gt_height"] is not None
                        else None
                    ),
                    "width_error_m": (
                        round(result["width_error_m"], 4)
                        if result["width_error_m"] is not None
                        else None
                    ),
                    "height_error_m": (
                        round(result["height_error_m"], 4)
                        if result["height_error_m"] is not None
                        else None
                    ),
                    "decision":           decision,
                    "gt_decision":        gt_dec,
                    "correct":            correct,
                    "timestamp":          datetime.now().isoformat()
                }

                self.rows.append(row)
                # NEW: successful trial — failure_type stays "" (blank),
                # not one of the fixed codes, since nothing failed.
                row_v2["failure_type"] = ""
                self.rows_v2.append(row_v2)
                if correct is not None:
                    total_correct    += int(correct)
                    evaluated_trials += 1
                total_trials  += 1

                gt_width_str = (
                    f"{result['gt_width']:.3f}"
                    if result["gt_width"] is not None
                    else "N/A"
                )

                log.info(
                    f"[{obj_name} | trial {trial:02d}/{self.n_trials:02d} | "
                    f"{'OK' if correct else 'X'}] "
                    f"W_est={result['estimated_width']:.3f} "
                    f"W_gt={gt_width_str} | {decision}"
                )

                time.sleep(0.5)   # small pause between trials

        # ── Write CSV ─────────────────────────────────────────────────────
        self._write_csv()
        self._write_csv_v2()

        estimator.destroy_node()

        # ── Print summary ─────────────────────────────────────────────────
        accuracy = total_correct / evaluated_trials * 100 if evaluated_trials > 0 else 0
        log.info("=" * 50)
        log.info(f"EXPERIMENT COMPLETE — {self.condition} condition "
                  f"(scene: {self.scene_name})")
        log.info(f"Accuracy: {total_correct}/{evaluated_trials} correct ({accuracy:.1f}%) "
                  f"[{total_trials} total trials]")
        log.info(f"Results saved to: {self.csv_path}")
        log.info("=" * 50)

    def _write_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerows(self.rows)
        self.node.get_logger().info(f"CSV written: {self.csv_path}")

    def _write_csv_v2(self):
        """NEW: write the new pilot-experiment schema to its own file."""
        for row in self.rows_v2:
            assert row["failure_type"] in FAILURE_CODES or row["failure_type"] == "", (
                f"failure_type must be '' or one of {FAILURE_CODES}, "
                f"got {row['failure_type']!r}"
            )
        with open(self.csv_path_v2, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS_V2)
            writer.writeheader()
            writer.writerows(self.rows_v2)
        self.node.get_logger().info(f"CSV (v2 schema) written: {self.csv_path_v2}")


# ── Analysis: compare baseline vs proposed ───────────────────────────────
def analyse_results(output_dir: str):
    """
    Load all CSVs from output_dir, compute summary statistics,
    and print a comparison table for the paper.
    """
    csv_files = glob.glob(os.path.join(output_dir, "results_*.csv"))
    if not csv_files:
        print("No results found in", output_dir)
        return

    data = collections.defaultdict(list)

    for path in csv_files:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[row["condition"]].append(row)

    print("\n" + "=" * 60)
    print("GRASP ESTIMATION RESULTS — COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Metric':<35} {'Baseline':>10} {'Proposed':>10}")
    print("-" * 60)

    for condition in ["baseline", "proposed"]:
        rows = data.get(condition, [])
        if not rows:
            print(f"  No data found for condition: {condition}")
            continue

        evaluated = [r for r in rows if r["correct"] in ("True", "False")]
        correct       = [r for r in evaluated if r["correct"] == "True"]
        evaluated_trials = len(evaluated)
        accuracy      = (len(correct) / evaluated_trials * 100) if evaluated_trials > 0 else 0.0
        width_errors = [
            float(r["width_error_m"])
            for r in rows
            if r["width_error_m"] not in ("", "None", None, "nan")
        ]

        height_errors = [
            float(r["height_error_m"])
            for r in rows
            if r["height_error_m"] not in ("", "None", None, "nan")
        ]

        mean_w_err = (
            sum(width_errors) / len(width_errors)
            if width_errors else None
        )

        mean_h_err = (
            sum(height_errors) / len(height_errors)
            if height_errors else None
        )

        data[condition + "_stats"] = {
            "accuracy": accuracy,
            "mean_w_err": mean_w_err,
            "mean_h_err": mean_h_err,
            "n_trials": len(rows),
        }

    metrics = [
        ("Grasp accuracy (%)",          "accuracy",   ".1f"),
        ("Mean width error (m)",         "mean_w_err", ".4f"),
        ("Mean height error (m)",        "mean_h_err", ".4f"),
        ("Total trials",                 "n_trials",   "d"),
    ]

    for label, key, fmt in metrics:
        b = data.get("baseline_stats", {}).get(key, "N/A")
        p = data.get("proposed_stats", {}).get(key, "N/A")
        b_str = format(b, fmt) if isinstance(b, (int, float)) else str(b)
        p_str = format(p, fmt) if isinstance(p, (int, float)) else str(p)
        print(f"  {label:<33} {b_str:>10} {p_str:>10}")

    print("=" * 60)

    # Write summary CSV
    summary_path = SUMMARY_FILE.format(output_dir=output_dir)
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "baseline", "proposed"])
        for label, key, fmt in metrics:
            b = data.get("baseline_stats", {}).get(key, "N/A")
            p = data.get("proposed_stats", {}).get(key, "N/A")
            writer.writerow([label, b, p])

    print(f"\nSummary CSV written to: {summary_path}")


# ── Entry point ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, default=None,
                        help="Scene to run (e.g. world3_kitchen_objects). "
                             "Required unless --analyse.")
    parser.add_argument("--condition", choices=["baseline", "proposed"],
                        help="Which model condition to run")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of trials per object (default: 20)")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to read/write result CSVs")
    parser.add_argument("--analyse", action="store_true",
                        help="Analyse existing results and print comparison table")
    # Allow ROS args (--ros-args ...) to pass through without argparse choking
    args, _ = parser.parse_known_args()

    if args.analyse:
        analyse_results(args.results_dir)
        return

    if not args.condition:
        parser.error("--condition is required unless using --analyse")
    if not args.scene_name:
        parser.error("--scene_name is required unless using --analyse")

    rclpy.init(args=sys.argv)
    node = rclpy.create_node("results_logger")

    # Allow scene_name / condition / trials / results_dir to also be set
    # as ROS 2 parameters (e.g. via --ros-args -p scene_name:=...).
    node.declare_parameter("scene_name", args.scene_name)
    node.declare_parameter("condition", args.condition)
    node.declare_parameter("trials", args.trials)
    node.declare_parameter("results_dir", args.results_dir)
    node.declare_parameter("require_fresh_cloud", True)

    scene_name  = node.get_parameter("scene_name").get_parameter_value().string_value
    condition   = node.get_parameter("condition").get_parameter_value().string_value
    trials      = node.get_parameter("trials").get_parameter_value().integer_value
    results_dir = node.get_parameter("results_dir").get_parameter_value().string_value
    require_fresh_cloud = node.get_parameter("require_fresh_cloud").get_parameter_value().bool_value

    try:
        logger = ResultsLogger(
            node=node,
            scene_name=scene_name,
            condition=condition,
            n_trials=trials,
            output_dir=results_dir,
            require_fresh_cloud=require_fresh_cloud,
        )
        logger.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()