"""
conformal_gate.py

Purpose
-------
Split conformal prediction gate. Calibrates a threshold on a nonconformity
score using SUCCESSFUL trials only, then flags future trials whose score
exceeds that threshold as "unsafe to proceed." Distribution-free, finite-
sample coverage guarantee -- no assumption about the score's distribution,
only that calibration and test data are exchangeable.

Nonconformity score (per PREREG.md C4):
    s = Gap_initial / percentile_90(calibration_gaps)

Guarantee: if calibration and test data are exchangeable, then on a fresh
test point drawn from the same distribution as calibration,
    P(s_test > threshold) <= alpha
where threshold is the (1-alpha)-quantile of calibration scores (with the
standard finite-sample correction below).

Edge cases handled:
  - Finite-sample quantile correction: uses ceil((n+1)*(1-alpha))/n, not the
    naive percentile, since the naive version under-covers at finite n.
  - Calibration set must be successful trials only (per PREREG.md) -- this
    is enforced by requiring the caller to pre-filter, but this module
    additionally asserts on it defensively (checks a 'failure' column if
    present and errors loudly if any calibration row is a failure, rather
    than silently including it and quietly breaking the guarantee).
  - percentile_90(calibration_gaps) = 0 (degenerate, e.g. all Gaps are 0):
    guarded against divide-by-zero, raises explicitly rather than emitting
    inf/nan scores silently.
  - Per-object coverage check: the gate is fit pooled but coverage is
    ALSO validated per object, since pooled coverage can hide an object
    where exchangeability is violated (this is a pre-registered check,
    not an afterthought).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConformalGateResult:
    threshold: float
    scale: float  # the percentile_90(calibration_gaps) denominator
    alpha: float
    n_calibration: int


class ConformalGate:
    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self._fitted: Optional[ConformalGateResult] = None

    def _nonconformity_score(self, gap_values: np.ndarray, scale: float) -> np.ndarray:
        return gap_values / scale

    def fit(self, calibration_df: pd.DataFrame, gap_col: str = "Gap_initial"):
        """Calibrate the threshold on successful trials only.

        Raises
        ------
        ValueError if calibration_df contains any failed trials (if a
        'failure' column is present) -- this is the split-discipline rule
        from PREREG.md enforced in code, not just in the write-up.
        """
        if "failure" in calibration_df.columns and calibration_df["failure"].sum() > 0:
            raise ValueError(
                "Calibration set must contain successful trials only "
                "(per PREREG.md C4 split-discipline rule). "
                f"Found {calibration_df['failure'].sum()} failed trials in the "
                f"calibration set -- filter these out before calling fit()."
            )

        gaps = calibration_df[gap_col].values
        scale = np.percentile(gaps, 90)
        if scale <= 0:
            raise ValueError(
                f"percentile_90({gap_col}) on the calibration set is {scale} "
                f"(<=0) -- cannot form a nonconformity score by dividing by "
                f"this. Check for a degenerate calibration set (e.g. all "
                f"Gap values are exactly 0)."
            )

        scores = self._nonconformity_score(gaps, scale)
        n = len(scores)
        # finite-sample-corrected quantile (Vovk et al.): use the
        # ceil((n+1)(1-alpha))/n quantile, clipped to 1.0, not the naive
        # (1-alpha) percentile -- the naive version under-covers at finite n.
        q_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        threshold = float(np.quantile(scores, q_level, method="higher"))

        self._fitted = ConformalGateResult(
            threshold=threshold, scale=float(scale), alpha=self.alpha, n_calibration=n
        )
        return self

    def flag(self, test_df: pd.DataFrame, gap_col: str = "Gap_initial") -> np.ndarray:
        """Returns a boolean array: True = gate fires (flagged as unsafe)."""
        if self._fitted is None:
            raise RuntimeError("Call fit() before flag().")
        scores = self._nonconformity_score(test_df[gap_col].values, self._fitted.scale)
        return scores > self._fitted.threshold

    def empirical_coverage(self, test_df: pd.DataFrame, gap_col: str = "Gap_initial") -> dict:
        """Empirical coverage on a test set: fraction of test points NOT
        flagged (i.e. the gate correctly lets successful-looking trials
        through). Reports pooled and, if object_id is present, per-object."""
        if self._fitted is None:
            raise RuntimeError("Call fit() before empirical_coverage().")

        flagged = self.flag(test_df, gap_col)
        pooled_flag_rate = float(flagged.mean())
        pooled_coverage = 1.0 - pooled_flag_rate

        result = {
            "pooled_flag_rate": pooled_flag_rate,
            "pooled_coverage": pooled_coverage,
            "target_coverage": 1 - self.alpha,
            "within_5pp": abs(pooled_coverage - (1 - self.alpha)) <= 0.05,
            "per_object": None,
        }

        if "object_id" in test_df.columns:
            per_obj_rows = []
            for obj, sub in test_df.groupby("object_id"):
                obj_flagged = self.flag(sub, gap_col)
                obj_coverage = 1.0 - obj_flagged.mean()
                per_obj_rows.append({
                    "object_id": obj,
                    "n": len(sub),
                    "flag_rate": float(obj_flagged.mean()),
                    "coverage": float(obj_coverage),
                    "within_5pp_of_target": abs(obj_coverage - (1 - self.alpha)) <= 0.05,
                })
            result["per_object"] = pd.DataFrame(per_obj_rows)

        return result


def three_split(df: pd.DataFrame, calib_frac: float = 0.7, seed: int = 42):
    """Split into calibration (successful trials only, per PREREG.md) and
    test (all trials, both outcomes -- coverage is about correctly passing
    successful trials while a realistic test set still contains failures).

    Edge case: if calib_frac would leave too few successful trials to
    calibrate reliably (n<20), warns rather than silently proceeding blind.
    """
    rng = np.random.default_rng(seed)
    success_df = df[df["failure"] == 0].reset_index(drop=True)
    idx = rng.permutation(len(success_df))
    n_calib = int(len(success_df) * calib_frac)
    calib_idx = idx[:n_calib]
    calib_df = success_df.iloc[calib_idx].reset_index(drop=True)

    # test set = remaining successful trials + ALL failed trials (realistic
    # test conditions include both outcomes)
    remaining_success = success_df.iloc[idx[n_calib:]].reset_index(drop=True)
    failed_df = df[df["failure"] == 1].reset_index(drop=True)
    test_df = pd.concat([remaining_success, failed_df], ignore_index=True)

    if len(calib_df) < 20:
        print(f"  WARNING: only {len(calib_df)} successful trials in calibration "
              f"set -- conformal threshold may be unstable at this N.")

    return calib_df, test_df


if __name__ == "__main__":
    import os
    mock_path = os.path.join(os.path.dirname(__file__), "data", "mock_trials.csv")
    if not os.path.exists(mock_path):
        raise FileNotFoundError("Run mock_data_generator.py first.")
    df = pd.read_csv(mock_path)

    calib_df, test_df = three_split(df, calib_frac=0.7)
    print(f"Calibration set (successful only): {len(calib_df)} trials")
    print(f"Test set (mixed outcomes): {len(test_df)} trials, "
          f"{test_df['failure'].sum()} failures")

    gate = ConformalGate(alpha=0.10)
    gate.fit(calib_df)
    print(f"\nFitted threshold: {gate._fitted.threshold:.4f} "
          f"(scale={gate._fitted.scale:.4f}, n_calib={gate._fitted.n_calibration})")

    cov = gate.empirical_coverage(test_df)
    print(f"\nPooled coverage: {cov['pooled_coverage']:.3f} "
          f"(target: {cov['target_coverage']:.3f}) "
          f"{'PASS (within 5pp)' if cov['within_5pp'] else 'FAIL (outside 5pp)'}")
    print(f"Flag rate: {cov['pooled_flag_rate']:.3f} "
          f"(healthy range: 15-40%; {'OK' if 0.15 <= cov['pooled_flag_rate'] <= 0.40 else 'outside healthy range -- check'})")

    print("\nPer-object coverage:")
    print(cov["per_object"].to_string(index=False))