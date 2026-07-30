"""
dry_run_e2e.py -- Week 2, Day 4: full synthetic end-to-end pipeline wiring.

Wires: synthetic camera frames -> occlusion_estimator.py (true_visibility,
independent of YOLO) -> tsdf_fusion.py (perceived_visibility) ->
gap_computer.py (Gap features) -> logger.py (JSONL trial records) ->
a quick logistic-regression smoke check (proxy for analysis_c1.py).

This is the first time these pieces run as ONE continuous loop rather than
isolated, individually-self-tested units. Bugs this catches that unit tests
can't: field-name mismatches between modules, unit mismatches (mm vs m),
and schema drift between logger.py and what analysis scripts expect.

No real YOLO detector exists yet -- 'perceived visibility' here comes from
tsdf_fusion fed a synthetic mask standing in for a YOLO detection, exactly
like tsdf_fusion.py's own self-test does. This is a wiring/mechanics test,
not a claim about real detector behavior.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gazebo"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mock-pipeline"))

from occlusion_estimator import ObjectGeometry, Pose, compute_true_visibility
from tsdf_fusion import CameraIntrinsics, MaskedTSDFFusion, make_synthetic_depth_frame
from gap_computer import compute_gap_features
from logger import TrialRecord, TrialLogger

OBJECTS = {
    "apple":  ObjectGeometry(width=0.08, height=0.08, depth=0.08),
    "remote": ObjectGeometry(width=0.05, height=0.15, depth=0.02),
    "glass":  ObjectGeometry(width=0.06, height=0.15, depth=0.06,
                              marker_to_object_offset=(0.05, 0.0, 0.0)),
}
INTRINSICS_GEOM = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
INTRINSICS_TSDF = CameraIntrinsics(width=640, height=480, fx=500, fy=500, cx=320, cy=240)
OCCLUDER_GEOM_TRUTH = ObjectGeometry(width=0.20, height=0.30, depth=0.02)

LOG_PATH = "dry_run_data/dry_run_trials.jsonl"


def run_one_trial(trial_id, session_id, object_id, occluder_z, clutter_count, rng):
    geom = OBJECTS[object_id]
    target_pose = Pose(x=0.0, y=0.0, z=0.5)
    occluder_pose = Pose(x=0.0, y=0.0, z=occluder_z)

    # 1. Ground truth (occlusion_estimator -- must stay YOLO-independent)
    true_vis = compute_true_visibility(target_pose, geom, occluder_pose,
                                        OCCLUDER_GEOM_TRUTH, INTRINSICS_GEOM)

    # 2. Perceived visibility (tsdf_fusion, fed a synthetic mask standing in
    #    for a YOLO detection -- occluder shrinks the mask/depth region)
    fusion = MaskedTSDFFusion(voxel_length=0.004, sdf_trunc=0.02)
    target_center = np.array([0.0, 0.0, 0.5])
    target_half = np.array([geom.width/2, geom.height/2, geom.depth/2])
    occ_frac_px = int(320 * (1.0 - true_vis))  # crude px-based occlusion proxy
    occluder_uv_box = (280, 0, 280 + max(occ_frac_px, 0), 480) if true_vis < 1.0 else None
    color, depth, mask = make_synthetic_depth_frame(
        640, 480, INTRINSICS_TSDF, target_center, target_half,
        occluder_uv_box=occluder_uv_box, dropout_fraction=0.0, seed=trial_id)
    fusion.integrate_frame(color, depth, mask, INTRINSICS_TSDF)
    perceived_vis = fusion.compute_perceived_visibility(target_center, target_half, grid_resolution=6)

    # add small independent detector noise so perceived != a deterministic
    # function of true_vis alone (mirrors real YOLO+TSDF noise)
    perceived_vis = float(np.clip(perceived_vis + rng.normal(0, 0.03), 0, 1))

    # 3. Gap features (single-step trial here -> slope=0, that's expected)
    gap = compute_gap_features([perceived_vis], [true_vis])

    # 4. Outcome model (placeholder failure rule, NOT the real study's model
    #    -- just needs to produce a non-degenerate 0/1 for the wiring test)
    p_fail = np.clip(0.1 + 0.6 * (1 - true_vis) + 1.5 * gap.gap_initial, 0, 0.95)
    failure = int(rng.random() < p_fail)

    return TrialRecord(
        trial_id=trial_id, session_id=session_id, object_id=object_id,
        session_baseline_failure_rate=0.05,
        measured_true_visibility=true_vis, true_occlusion=1 - true_vis,
        clutter_count=clutter_count, perceived_visibility_initial=perceived_vis,
        Gap_initial=gap.gap_initial, Gap_max=gap.gap_max, Gap_slope=gap.gap_slope,
        failure=failure, camera_pose_variant="front_0deg",
    )


def main():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    frozen_marker = LOG_PATH + ".FROZEN"
    if os.path.exists(frozen_marker):
        os.remove(frozen_marker)

    logger = TrialLogger(LOG_PATH)
    rng = np.random.default_rng(0)

    print("=" * 70)
    print("Week 2 Day 4: Full synthetic end-to-end dry run")
    print("=" * 70)

    trial_id = 0
    occluder_z_levels = [0.35, 0.4, 0.45, 0.55]  # varying occlusion severity
    for session_id, obj in enumerate(OBJECTS):
        for occ_z in occluder_z_levels:
            for clutter in [0, 2]:
                record = run_one_trial(trial_id, session_id, obj, occ_z, clutter, rng)
                logger.append_trial(record)
                trial_id += 1

    print(f"\n{trial_id} synthetic trials run through the FULL pipeline "
          f"(occlusion_estimator -> tsdf_fusion -> gap_computer -> logger).")

    # Reload from disk exactly as a real analysis script would
    trials = logger.load_trials()
    df = pd.DataFrame(trials)
    print(f"Reloaded {len(df)} trials from {LOG_PATH}.")
    print(f"\nSchema check -- columns present:\n  {list(df.columns)}")

    print(f"\nSanity ranges:")
    for col in ["measured_true_visibility", "perceived_visibility_initial", "Gap_initial"]:
        print(f"  {col}: min={df[col].min():.3f}, max={df[col].max():.3f}, mean={df[col].mean():.3f}")

    print(f"\nFailure rate by object:")
    print(df.groupby("object_id")["failure"].mean())

    # Quick logistic smoke check (stand-in for analysis_c1.py -- confirms the
    # frozen JSONL schema is directly usable by a statsmodels formula call,
    # which is the actual integration risk this dry run exists to catch)
    import statsmodels.formula.api as smf
    try:
        model = smf.logit("failure ~ true_occlusion + Gap_initial + C(object_id)", data=df).fit(disp=0)
        print(f"\nSmoke-check logistic fit succeeded (schema is analysis-ready).")
        print(f"Gap_initial coef={model.params.get('Gap_initial', float('nan')):.3f}, "
              f"p={model.pvalues.get('Gap_initial', float('nan')):.3f}")
    except Exception as e:
        print(f"\nSmoke-check logistic fit FAILED: {e}")
        print("This means the logger schema and analysis script formula are NOT "
              "yet compatible -- fix before trusting the real pipeline.")

    logger.freeze()
    print(f"\nDataset frozen at {frozen_marker} (dry-run discipline check -- "
          f"confirms freeze() blocks further appends).")
    try:
        logger.append_trial(trials_record_dummy := run_one_trial(9999, 0, "apple", 0.4, 0, rng))
        print("ERROR: append after freeze should have raised, but did not.")
    except RuntimeError:
        print("Freeze correctly blocks further appends -- PASS.")

    print("\n" + "=" * 70)
    print("Day 4 dry run complete: full loop wired, ran, and reloaded without error.")
    print("=" * 70)


if __name__ == "__main__":
    main()