"""
mock_data_generator.py

Purpose
-------
Generate FABRICATED trial records with a KNOWN, hand-specified relationship
between Gap and failure. This is not real data and is never reported as a
result. Its only job: let analysis_c1.py / analysis_c4.py be tested against
data where we already know the right answer, so we trust the scripts before
pointing them at real trial logs.

Ground truth we bake in (write this down BEFORE looking at any script output):
    P(failure) = sigmoid(
          b0
        + b_occ   * true_occlusion
        + b_gap   * Gap_initial
        + b_obj[object_id]      # per-object intercept shift, small
    )
  with b_gap > 0 and b_occ > 0, i.e. both occlusion and Gap independently
  raise failure probability. object_id shifts baseline difficulty slightly
  but does NOT change the sign or rough magnitude of b_gap -- that's the
  "generalizes across objects" property analysis_c1 should recover.

How to check this file did its job (see SUMMARY.md "How to check" for c1/c4):
  - Fit the same logistic form on the output and confirm the recovered
    Gap coefficient is positive and roughly in the ballpark of b_gap below.
  - Per-object coefficient sign should be positive for every object, even
    though per-object significance may not clear p<0.05 at N=30-50/object.

Edge cases handled:
  - Clamps true_occlusion, perceived_visibility, true_visibility to [0, 1].
  - Gap_max >= Gap_initial by construction (max over a monotonically-ish
    noisy trajectory that starts at Gap_initial).
  - Session baseline failure rate is injected as a small per-session offset,
    independent of Gap, so downstream code that controls for it has
    something real to control for.
  - Uneven per-object N is supported (pass a dict), mirroring the real
    study's "must-have vs nice-to-have" object split.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42

# Ground-truth coefficients (edit here only, never post-hoc)
TRUE_COEFS = {
    "b0": -2.2,        # baseline log-odds of failure at zero occlusion/Gap
    "b_occ": 2.0,      # occlusion's effect on failure
    "b_gap": 2.5,      # Gap's effect on failure -- this is what C1 should recover
    "b_session": 0.8,  # scale of per-session baseline-failure-rate offset
}

OBJECTS = ["apple", "purse", "glass"]

OBJECT_INTERCEPT_SHIFT = {
    "apple": 0.0,
    "purse": -0.2,
    "glass": 0.35,   # hardest to detect (transparent/reflective) -> highest baseline difficulty
}

OBJECT_DETECTOR_NOISE = {
    "apple": 0.03,
    "purse": 0.07,
    "glass": 0.12,
}

# 75-90 trials total per master doc Section 6/Week 5, ~25-30/object, even split
# (paper doesn't specify a must-have/nice-to-have split at 3 objects -- all 3 are must-have)
DEFAULT_N_PER_OBJECT = {"apple": 30, "purse": 30, "glass": 30}

N_SESSIONS = 8


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_mock_trials(n_per_object=None, seed=RNG_SEED, gap_signal_multiplier=1.0):
    """Generate a synthetic trial-level DataFrame with known Gap-failure structure.

    Parameters
    ----------
    n_per_object : dict[str, int] or None
        Trials per object. Defaults to DEFAULT_N_PER_OBJECT (uneven, mirrors
        the must-have/nice-to-have split in PREREG.md).
    seed : int
        RNG seed. Fixed by default so repeated runs are reproducible --
        mirrors the "frozen seed" discipline used elsewhere in the project.

    Returns
    -------
    pd.DataFrame with columns matching the real logger.py schema:
        trial_id, session_id, object_id, session_baseline_failure_rate,
        true_occlusion, clutter_count, perceived_visibility_initial,
        true_visibility, Gap_initial, Gap_max, Gap_slope, failure
    """
    n_per_object = n_per_object or DEFAULT_N_PER_OBJECT
    rng = np.random.default_rng(seed)

    rows = []
    trial_id = 0
    # assign a session baseline offset per session, shared across objects
    # run within it (mirrors "block sessions by object" but sessions still
    # span multiple days/objects overall)
    session_offsets = {s: rng.normal(0, 1) for s in range(N_SESSIONS)}

    for obj in OBJECTS:
        n = n_per_object.get(obj, 0)
        if n <= 0:
            continue
        for i in range(n):
            session_id = int(rng.integers(0, N_SESSIONS))
            session_offset = session_offsets[session_id]
            session_baseline_failure_rate = float(np.clip(
                _sigmoid(session_offset * TRUE_COEFS["b_session"]), 0.0, 1.0
            ))

            true_occlusion = float(np.clip(rng.beta(2, 2), 0, 1))
            clutter_count = int(rng.integers(0, 6))

            true_visibility = float(np.clip(1.0 - true_occlusion + rng.normal(0, 0.03), 0, 1))

            # perceived_visibility is a noisy read of true_visibility; noise
            # scale depends on the object's detectability (glass/remote noisier)
            noise_scale = OBJECT_DETECTOR_NOISE[obj] * gap_signal_multiplier
            perceived_visibility_initial = float(np.clip(
                true_visibility + rng.normal(0, noise_scale), 0, 1
            ))

            gap_initial = abs(perceived_visibility_initial - true_visibility)
            # Gap can drift upward slightly over a trial (max >= initial by construction)
            gap_drift = abs(rng.normal(0, noise_scale * 0.5))
            gap_max = gap_initial + gap_drift
            gap_slope = gap_drift / max(1, rng.integers(3, 10))  # rough per-step slope

            logit = (
                TRUE_COEFS["b0"]
                + TRUE_COEFS["b_occ"] * true_occlusion
                + TRUE_COEFS["b_gap"] * gap_signal_multiplier * gap_initial
                + OBJECT_INTERCEPT_SHIFT[obj]
                + session_offset * TRUE_COEFS["b_session"] * 0.3
            )
            p_fail = _sigmoid(logit)
            failure = int(rng.random() < p_fail)

            rows.append({
                "trial_id": trial_id,
                "session_id": session_id,
                "object_id": obj,
                "session_baseline_failure_rate": session_baseline_failure_rate,
                "true_occlusion": true_occlusion,
                "clutter_count": clutter_count,
                "perceived_visibility_initial": perceived_visibility_initial,
                "true_visibility": true_visibility,
                "Gap_initial": gap_initial,
                "Gap_max": gap_max,
                "Gap_slope": gap_slope,
                "failure": failure,
            })
            trial_id += 1

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    import os
    os.makedirs("mock-pipeline/data", exist_ok=True)

    # Realistic-magnitude dataset: matches expected real-study detector noise
    # and effect sizes. Use this to get an honest preview of statistical power.
    df = generate_mock_trials()
    out_path = "mock-pipeline/data/mock_trials.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} REALISTIC-magnitude mock trials -> {out_path}")
    print(df.groupby("object_id")["failure"].agg(["count", "mean"]))

    # Positive-control dataset: exaggerated Gap signal, used ONLY to verify
    # analysis_c1.py's mechanics correctly recover a known strong effect.
    # Never use this to make claims about real effect sizes.
    df_strong = generate_mock_trials(gap_signal_multiplier=8.0)
    strong_path = "mock-pipeline/data/mock_trials_positive_control.csv"
    df_strong.to_csv(strong_path, index=False)
    print(f"\nGenerated {len(df_strong)} POSITIVE-CONTROL mock trials (8x Gap signal) -> {strong_path}")
    print(df_strong.groupby("object_id")["failure"].agg(["count", "mean"]))

    print("\nGround truth coefficients used (for verification):", TRUE_COEFS)