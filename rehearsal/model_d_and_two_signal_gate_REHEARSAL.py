#!/usr/bin/env python3
"""
rehearsal/model_d_and_two_signal_gate_REHEARSAL.py

SYNTHETIC WIRING REHEARSAL -- NOT A RESULT.

T4: proves the statistical code path for PREREG.md's Amendment
2026-07-14 dual-signal extension (Model D and the two-signal conformal
gate) runs end-to-end using REAL, non-stub prg_edge_error values (from
T3's diagnosis/prg_edge_error_sweep_SIM.csv, 43 rows across all 6 staged
worlds), combined with SYNTHETIC Gap/occlusion/failure columns (there is
no real trial data of any kind -- see PROJECT_STATUS.md sec. 4).

Does NOT modify conformal_gate.py, gap_computer.py, or any of Person A's
files -- this is a new script, following daily_integration_check.py's own
existing labeling discipline. Every printed number here is a wiring
check, never a statistical finding, never citable as C1/C1-ext/C4/C4-ext
evidence.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRG_SWEEP_CSV = PROJECT_ROOT / "diagnosis" / "prg_edge_error_sweep_SIM.csv"
SEED = 42


def build_synthetic_trial_table(prg_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """One synthetic 'trial' per real prg_edge_error row. Gap/occlusion/
    failure are SYNTHETIC, generated with a deliberate (fake, for wiring
    purposes only) positive relationship to prg_edge_error and occlusion,
    so Model D's coefficients aren't just noise -- this is NOT a claim
    about the real relationship, only a check that the code path can
    recover a signal that was deliberately planted."""
    rng = np.random.default_rng(seed)
    n = len(prg_df)

    # Grouped by world (6 groups), not by individual object_a (19 unique
    # values across only 43 rows) -- the fine-grained grouping produced a
    # singular design matrix (near one dummy per 2 rows). World-level
    # grouping is still a real fixed-effect code-path exercise, just at a
    # cardinality this small synthetic sample can actually support.
    object_id = prg_df["world"].to_numpy()
    true_occlusion = rng.uniform(0.0, 0.8, size=n)
    gap_initial = np.clip(
        0.3 * true_occlusion + 0.2 * rng.normal(size=n) + 0.05, 0.0, 1.0
    )
    prg_edge_error = prg_df["prg_edge_error"].to_numpy()

    # Planted synthetic failure model: higher occlusion, Gap, and
    # prg_edge_error each independently push failure probability up.
    # Coefficients are arbitrary, chosen only so the fitted model has
    # something non-degenerate to recover -- not calibrated to anything.
    logit = -1.5 + 2.0 * true_occlusion + 2.5 * gap_initial + 1.5 * prg_edge_error
    p_failure = 1.0 / (1.0 + np.exp(-logit))
    failure = (rng.uniform(size=n) < p_failure).astype(int)

    return pd.DataFrame({
        "object_id": object_id,
        "true_occlusion": true_occlusion,
        "Gap_initial": gap_initial,
        "prg_edge_error": prg_edge_error,
        "failure": failure,
    })


def fit_model_d(df: pd.DataFrame):
    """failure ~ true_occlusion + Gap_initial + prg_edge_error + C(object_id)"""
    model = smf.logit(
        "failure ~ true_occlusion + Gap_initial + prg_edge_error + C(object_id)",
        data=df,
    )
    return model.fit(disp=False)


def two_signal_conformal_gate(calib_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float = 0.10):
    """Max-combination nonconformity score over Gap and prg_edge_error,
    each independently scaled by its own calibration-set 90th percentile
    (matching conformal_gate.py's existing single-signal scale
    convention), then combined via max() so either signal alone can
    trigger the gate."""
    gap_scale = np.percentile(calib_df["Gap_initial"], 90)
    prg_scale = np.percentile(calib_df["prg_edge_error"], 90)
    if gap_scale <= 0 or prg_scale <= 0:
        raise ValueError(f"Degenerate calibration scale: gap_scale={gap_scale}, prg_scale={prg_scale}")

    def score(df):
        s_gap = df["Gap_initial"].to_numpy() / gap_scale
        s_prg = df["prg_edge_error"].to_numpy() / prg_scale
        return np.maximum(s_gap, s_prg)

    calib_scores = score(calib_df)
    n = len(calib_scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    threshold = float(np.quantile(calib_scores, q_level, method="higher"))

    test_scores = score(test_df)
    flagged = test_scores > threshold
    coverage = 1.0 - flagged.mean()
    return {
        "threshold": threshold, "gap_scale": gap_scale, "prg_scale": prg_scale,
        "n_calibration": n, "flag_rate": float(flagged.mean()), "coverage": float(coverage),
        "target_coverage": 1 - alpha,
    }


def single_signal_gate_gap_only(calib_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float = 0.10):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "mock-pipeline"))
    from conformal_gate import ConformalGate
    gate = ConformalGate(alpha=alpha)
    gate.fit(calib_df, gap_col="Gap_initial")
    return gate.empirical_coverage(test_df, gap_col="Gap_initial")


def main():
    print("=" * 78)
    print("SYNTHETIC WIRING REHEARSAL -- NOT A RESULT")
    print("Model D + two-signal conformal gate, real prg_edge_error, synthetic")
    print("Gap/occlusion/failure. See PROJECT_STATUS.md sec 4/5 item 5.")
    print("=" * 78)

    if not PRG_SWEEP_CSV.exists():
        raise FileNotFoundError(
            f"{PRG_SWEEP_CSV} not found -- run "
            "diagnosis/scripts/sweep_prg_edge_error_SIM.py first (T3)."
        )
    prg_df = pd.read_csv(PRG_SWEEP_CSV)
    trial_df = build_synthetic_trial_table(prg_df)

    print(f"\nSynthetic trial table: {len(trial_df)} rows "
          f"(1 per real prg_edge_error value from T3's sweep)")
    print(f"Failure rate (synthetic, planted): {trial_df['failure'].mean():.3f}")

    print("\n--- Model D: failure ~ true_occlusion + Gap_initial + prg_edge_error + C(object_id) ---")
    result = fit_model_d(trial_df)
    print(result.summary())

    print("\n--- Two-signal conformal gate (max-combination, Gap + prg_edge_error) ---")
    success_df = trial_df[trial_df["failure"] == 0].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(success_df))
    n_calib = max(1, int(len(success_df) * 0.7))
    calib_df = success_df.iloc[idx[:n_calib]].reset_index(drop=True)
    remaining_success = success_df.iloc[idx[n_calib:]].reset_index(drop=True)
    failed_df = trial_df[trial_df["failure"] == 1].reset_index(drop=True)
    test_df = pd.concat([remaining_success, failed_df], ignore_index=True)

    print(f"Calibration (successful only): {len(calib_df)}; Test (mixed): {len(test_df)}")

    two_signal = two_signal_conformal_gate(calib_df, test_df)
    print(f"Two-signal gate: threshold={two_signal['threshold']:.4f}, "
          f"flag_rate={two_signal['flag_rate']:.3f}, "
          f"coverage={two_signal['coverage']:.3f} (target={two_signal['target_coverage']:.3f})")

    single_signal = single_signal_gate_gap_only(calib_df, test_df)
    print(f"Single-signal (Gap only) gate: flag_rate={single_signal['pooled_flag_rate']:.3f}, "
          f"coverage={single_signal['pooled_coverage']:.3f} "
          f"(target={single_signal['target_coverage']:.3f})")

    print("\n" + "=" * 78)
    print("REMINDER: every number above is a SYNTHETIC WIRING REHEARSAL. Gap,")
    print("true_occlusion, and failure are fabricated with a planted synthetic")
    print("relationship; only prg_edge_error is real (non-stub) sim data from")
    print("T3. This proves the Model D / two-signal-gate CODE PATH runs given")
    print("real PRG_edge_error values -- it is NOT a statistical result and")
    print("must never be cited as C1/C1-ext/C4/C4-ext evidence.")
    print("=" * 78)


if __name__ == "__main__":
    main()
