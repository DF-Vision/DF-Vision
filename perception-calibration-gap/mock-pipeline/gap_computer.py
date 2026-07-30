"""
gap_computer.py

Purpose
-------
Compute the Calibration Gap from a per-step visibility trajectory. This is
pure arithmetic -- it does not know or care whether perceived_visibility
came from a real YOLO+TSDF pipeline or a synthetic Gazebo/mock source. That
separation is deliberate: it's the "zero-dependency swap point" from the
project architecture -- this file never changes when you move from mock to
real hardware, only the upstream adapter producing the trajectory changes.

Definitions (must match PREREG.md exactly -- do not redefine silently):
    Gap(t)      = |perceived_visibility(t) - true_visibility(t)|
    Gap_initial = Gap(t=0)
    Gap_max     = max_t Gap(t)
    Gap_slope   = (Gap(t_end) - Gap(t=0)) / t_end   (simple end-to-end slope;
                  NOT a fitted regression slope -- keep it this simple and
                  say so in the methods section, don't silently upgrade it)

Edge cases handled:
  - Single-step trajectories (t_end == 0): Gap_slope defined as 0.0, not
    division by zero.
  - true_visibility must come from the marker/geometry pipeline, NEVER from
    YOLO -- this function takes it as a separate argument on purpose so it's
    structurally impossible to accidentally feed YOLO's own estimate into
    both sides of the Gap (which would silently make Gap measure YOLO
    against itself).
  - Values are not clamped here -- clamping happens upstream where
    perceived/true visibility are estimated. This function just takes the
    absolute difference at each step.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class GapResult:
    gap_initial: float
    gap_max: float
    gap_slope: float
    gap_trajectory: List[float]


def compute_gap_trajectory(perceived_visibility: List[float],
                            true_visibility: List[float]) -> List[float]:
    """Per-step |perceived - true| visibility gap.

    Both lists must be the same length and index-aligned to the same
    timesteps. Raises ValueError if not, rather than silently truncating --
    a silent length mismatch would quietly corrupt Gap_max/Gap_slope.
    """
    if len(perceived_visibility) != len(true_visibility):
        raise ValueError(
            f"perceived_visibility ({len(perceived_visibility)} steps) and "
            f"true_visibility ({len(true_visibility)} steps) must be the same "
            f"length and index-aligned."
        )
    if len(perceived_visibility) == 0:
        raise ValueError("Empty trajectory: at least one timestep is required.")

    return [abs(p - t) for p, t in zip(perceived_visibility, true_visibility)]


def compute_gap_features(perceived_visibility: List[float],
                          true_visibility: List[float]) -> GapResult:
    """Derive Gap_initial, Gap_max, Gap_slope from a full trial trajectory."""
    traj = compute_gap_trajectory(perceived_visibility, true_visibility)

    gap_initial = traj[0]
    gap_max = max(traj)

    t_end = len(traj) - 1
    if t_end == 0:
        gap_slope = 0.0  # single-step trial: slope undefined, not divide-by-zero
    else:
        gap_slope = (traj[-1] - traj[0]) / t_end

    return GapResult(
        gap_initial=gap_initial,
        gap_max=gap_max,
        gap_slope=gap_slope,
        gap_trajectory=traj,
    )


if __name__ == "__main__":
    # Self-test against hand-calculable cases before trusting this on real data.
    # Case 1: perfectly calibrated perception (Gap should be ~0 everywhere)
    perceived = [0.8, 0.8, 0.8]
    true = [0.8, 0.8, 0.8]
    r = compute_gap_features(perceived, true)
    assert abs(r.gap_initial) < 1e-9 and abs(r.gap_max) < 1e-9 and abs(r.gap_slope) < 1e-9
    print("Case 1 (perfect calibration) OK:", r)

    # Case 2: perception overestimates visibility, gap grows over time
    perceived = [0.9, 0.9, 0.9]
    true = [0.7, 0.6, 0.4]
    r = compute_gap_features(perceived, true)
    assert abs(r.gap_initial - 0.2) < 1e-9
    assert abs(r.gap_max - 0.5) < 1e-9
    assert abs(r.gap_slope - (0.5 - 0.2) / 2) < 1e-9
    print("Case 2 (growing gap) OK:", r)

    # Case 3: single-step trial -> slope must be 0.0, not an exception
    r = compute_gap_features([0.5], [0.3])
    assert r.gap_slope == 0.0
    print("Case 3 (single-step, no div-by-zero) OK:", r)

    # Case 4: mismatched lengths must raise, not silently truncate
    try:
        compute_gap_features([0.5, 0.4], [0.3])
        raise AssertionError("Expected ValueError for mismatched lengths")
    except ValueError as e:
        print("Case 4 (mismatched lengths correctly raises) OK:", e)

    print("\nAll gap_computer.py self-tests passed.")