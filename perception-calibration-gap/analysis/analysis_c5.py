"""
analysis_c5.py

Purpose
-------
Tests whether active re-observation (a second TSDF fusion viewpoint,
triggered when C4's gate flags a trial) measurably reduces Gap and recovers
some fraction of flagged trials from failure. Status: Primary (supporting
C4's practical value).

Pre-registered metrics (PREREG.md C5):
    - Recovery rate: fraction of C4-flagged trials where post-fusion Gap
      drops below the gate threshold AND the trial subsequently succeeds
    - Gap delta: Gap_after_fusion - Gap_before_fusion, as a full distribution
    - Negative-result tracking: trials where fusion INCREASES Gap or fails
      to prevent failure must be explicitly counted and reported, not
      filtered out (pre-registered requirement, not optional)

Pre-registered pass criterion: recovery rate > 0 and honestly reported --
no numeric threshold pre-committed (this is explicitly the most exploratory
of the 5 contributions), but the negative-result tracking below IS
mandatory regardless of how good the headline number looks.

How to check this script is correct:
    Run against mock data where a "post-fusion Gap" is simulated by
    combining the gate's flagged trials with tsdf_fusion.py's multi-frame
    re-fusion behavior (see mock_post_fusion_gap() below) -- the recovery
    rate should be nonzero but not 100% (100% would suggest an unrealistic
    mock setup, not a real result).

Edge cases handled:
    - Trials with NO flagged rows (e.g. a gate threshold set too high for
      this data) -- reported explicitly as "no flagged trials to evaluate",
      not silently returning an empty/misleading recovery rate of 0/0.
    - Gap_after_fusion is not guaranteed to improve -- explicitly tracked
      and counted, matching the pre-registered negative-result requirement.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mock-pipeline"))
from conformal_gate import ConformalGate, three_split  # noqa: E402


def mock_post_fusion_gap(flagged_df: pd.DataFrame, seed: int = 42,
                          improvement_prob: float = 0.65) -> pd.DataFrame:
    """Simulates the effect of an active re-observation step on Gap, for
    self-testing analysis_c5.py's mechanics only. Real data replaces this
    with an actual second-viewpoint TSDF fusion call (see tsdf_fusion.py's
    multi-frame re-fusion, already self-tested there).

    With probability improvement_prob, Gap decreases by a random amount
    (simulating a helpful re-observation); otherwise it stays flat or
    increases slightly (simulating fusion that doesn't help or actively
    hurts) -- this deliberately creates BOTH positive and negative cases,
    since a mock where fusion always helps would be a useless self-test.
    """
    rng = np.random.default_rng(seed)
    df = flagged_df.copy()
    n = len(df)
    improves = rng.random(n) < improvement_prob
    delta = np.where(
        improves,
        -np.abs(rng.normal(0.04, 0.02, n)),   # Gap decreases (helpful fusion)
        np.abs(rng.normal(0.01, 0.015, n)),    # Gap flat/slightly worse (unhelpful fusion)
    )
    df["Gap_after_fusion"] = np.clip(df["Gap_initial"] + delta, 0, 1)
    # Whether the trial subsequently succeeds: modeled as more likely if
    # Gap_after_fusion dropped meaningfully. This is a MOCK outcome model,
    # not a claim about real recovery mechanics.
    success_prob = np.clip(1.0 - df["Gap_after_fusion"] * 2.0, 0.05, 0.95)
    df["succeeded_after_fusion"] = (rng.random(n) < success_prob).astype(int)
    return df


def run_full_c5_package(df: pd.DataFrame, alpha: float = 0.10, seed: int = 42):
    print("=" * 70)
    print("C5: Active Mitigation via TSDF Re-Fusion (supporting C4's practical value)")
    print("=" * 70)

    calib_df, test_df = three_split(df, calib_frac=0.7, seed=seed)
    gate = ConformalGate(alpha=alpha)
    gate.fit(calib_df, gap_col="Gap_initial")

    flagged_mask = gate.flag(test_df, gap_col="Gap_initial")
    flagged_df = test_df[flagged_mask].copy()

    if len(flagged_df) == 0:
        print("\nNo flagged trials in the test set to evaluate C5 against -- "
              "reporting explicitly rather than a misleading 0/0 recovery rate. "
              "This can happen if the gate threshold is conservative or Gap "
              "values in this dataset rarely exceed it (see C4's flag-rate "
              "diagnostic for the same underlying issue).")
        return {"recovery_rate": None, "note": "no flagged trials"}

    print(f"\n{len(flagged_df)} trials flagged by C4's gate (threshold={gate._fitted.threshold:.4f})")

    post_fusion = mock_post_fusion_gap(flagged_df, seed=seed)

    gap_before = post_fusion["Gap_initial"].values
    gap_after = post_fusion["Gap_after_fusion"].values
    gap_delta = gap_after - gap_before

    print(f"\nGap delta (after - before fusion) distribution:")
    print(f"  mean={gap_delta.mean():.4f}, median={np.median(gap_delta):.4f}, "
          f"min={gap_delta.min():.4f}, max={gap_delta.max():.4f}")

    below_threshold_after = gap_after < gate._fitted.threshold
    succeeded = post_fusion["succeeded_after_fusion"].values.astype(bool)
    recovered = below_threshold_after & succeeded

    recovery_rate = float(recovered.mean())
    print(f"\nRecovery rate (Gap drops below gate threshold AND trial succeeds): "
          f"{recovery_rate:.3f} ({recovered.sum()}/{len(post_fusion)})")

    # Mandatory negative-result tracking -- never filtered out
    gap_increased = (gap_delta > 0).sum()
    fusion_failed_to_prevent_failure = ((~below_threshold_after) | (~succeeded)).sum()
    print(f"\nNegative-result tracking (mandatory, not optional):")
    print(f"  Trials where fusion INCREASED Gap: {gap_increased}/{len(post_fusion)} "
          f"({gap_increased/len(post_fusion)*100:.1f}%)")
    print(f"  Trials where fusion did NOT prevent failure / didn't drop below "
          f"threshold: {fusion_failed_to_prevent_failure}/{len(post_fusion)} "
          f"({fusion_failed_to_prevent_failure/len(post_fusion)*100:.1f}%)")

    print(f"\nPass criterion: recovery_rate > 0 and honestly reported -> "
          f"{'PASS' if recovery_rate > 0 else 'FAIL (recovery_rate is exactly 0)'}"
          f" (no further numeric threshold pre-committed for this exploratory contribution)")

    return {
        "recovery_rate": recovery_rate,
        "gap_delta_mean": float(gap_delta.mean()),
        "n_flagged": len(flagged_df),
        "n_gap_increased": int(gap_increased),
        "n_fusion_failed": int(fusion_failed_to_prevent_failure),
    }


if __name__ == "__main__":
    mock_path = os.path.join(os.path.dirname(__file__), "..", "mock-pipeline", "data", "mock_trials.csv")
    if not os.path.exists(mock_path):
        raise FileNotFoundError("Run mock_data_generator.py first.")
    df = pd.read_csv(mock_path)
    results = run_full_c5_package(df)