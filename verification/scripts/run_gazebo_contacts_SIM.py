#!/usr/bin/env python3
"""
verification/scripts/run_gazebo_contacts_SIM.py

V1f needs REAL, physics-verified gazebo_contact ground truth (not the
SDFGeometryRelationSource static-pose heuristic used as a stand-in last
session). relation_classifier/scripts/extract_gazebo_contacts.py exists
and implements exactly this, but its own docstring says it has NEVER been
executed ("This sandbox has no Gazebo install... none of the
gz.transport13/gz.msgs10 calls below have been run against a live
gz-sim process"). This environment DOES have `gz` (Gazebo Sim 8.14.0)
and importable gz.transport13/gz.msgs10 bindings, so this is a genuine,
bounded (rule 8: max 3 attempts) first-ever execution attempt.

Does NOT modify extract_gazebo_contacts.py -- imports process_world()
directly and points it at worlds/*.sdf (the dynamic, contact-sensor-
equipped world files used for real simulation), not
relation_classifier/training_worlds/*.sdf (extract_gazebo_contacts.py's
own hardcoded default, which is the STATIC/no-contact-sensor copy and
would silently produce nothing per the script's own documented static-
static contact-resolution dead end).

Headless simulation only (gz sim -s), no GUI, no hardware, no real robot.
Output labeled SIM.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))

import extract_gazebo_contacts as egc

OUT_DIR = PROJECT_ROOT / "verification" / "gazebo_contacts_SIM"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORLD_SDF_DIR = PROJECT_ROOT / "relation_classifier" / "training_worlds"
# CORRECTED after a first, wrong attempt against PROJECT_ROOT/"worlds": that
# directory (erf.WORLDS_SDF_DIR, used for pure-geometry ray-casting) is the
# STATIC, no-contact-sensor copy (verified: 47x <static>true</static>, 0x
# <static>false</static>, 0 <sensor type='contact'> blocks in
# world1_baseline.sdf). relation_classifier/training_worlds/ is the DYNAMIC,
# contact-sensor-equipped copy (35x static=true, 12x static=false, 12
# contact sensors) -- also extract_gazebo_contacts.py's own original
# hardcoded default (WORLDS_DIR = Path("../training_worlds")). The first
# attempt's "everything is unknown" result was fully explained by pointing
# at the wrong directory, not a Gazebo-build topic-naming issue -- see
# verification/V1_AUDIT.md for the full correction trail.


WORLD_FILES = [
    "world1_baseline.sdf", "world2_dense_clutter.sdf", "world3_thin_objects.sdf",
    "world4_support_scene.sdf", "world5_occlusion_scene.sdf", "world6_dense_mixed.sdf",
]


def run_one(world_file):
    sdf_path = WORLD_SDF_DIR / world_file
    if not sdf_path.exists():
        print(f"ERROR: {sdf_path} not found")
        return None, None, None
    print(f"\n{'='*70}\n{world_file}\n{'='*70}")
    pose_df, contact_df, flagged = egc.process_world(sdf_path)
    world_id = sdf_path.stem
    pose_df.to_csv(OUT_DIR / f"{world_id}_pose_delta_SIM.csv", index=False)
    contact_df.to_csv(OUT_DIR / f"{world_id}_gazebo_contacts_SIM.csv", index=False)
    n1 = int((contact_df["contact"] == 1).sum())
    nu = int((contact_df["contact"] == "unknown").sum())
    print(f"{world_file}: {len(contact_df)} pairs, contact=1: {n1}, unknown: {nu}, flagged: {flagged}")
    return pose_df, contact_df, flagged


def main():
    if len(sys.argv) > 1:
        targets = [sys.argv[1]]
    else:
        targets = WORLD_FILES
    for wf in targets:
        run_one(wf)


if __name__ == "__main__":
    main()
