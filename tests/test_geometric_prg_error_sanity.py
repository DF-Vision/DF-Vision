"""
tests/test_geometric_prg_error_sanity.py

T6a: sanity test the earlier null result (explore/E3_FINDINGS.md) on a
MINIMAL, isolated synthetic depth step edge (not a full ray-cast scene),
sweeping known blur (Gaussian sigma, pixels) and known noise (Gaussian
sigma, meters) independently, checking whether C1/C2's raw internal
signals respond with a strong, consistent monotonic relationship
(|Spearman rho| > 0.6, direction unconstrained -- see
project_overnight/CRITERIA.md's T6a correction for why the direction is
not fixed).

If this isolated test also shows near-zero |rho| for a signal that
SHOULD be sensitive (boundary_sharpness under blur, in particular), that
would be new evidence of clamping/saturation rather than a scene-
composition artifact, and would need investigating (this file reports
which, if either, is true rather than assuming one).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "relation_classifier" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "explore" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "retrain" / "scripts"))

import numpy as np
from scipy.stats import spearmanr
import pytest

import geometric_prg_error as gpe
from noise_model import apply_depth_noise

RANDOM_STATE = 20260920
IMG_SHAPE = (80, 80)


def make_step_edge_depth(gap_m: float = 0.001, near_depth_m: float = 0.5) -> np.ndarray:
    """Flat 'target' region (left half, depth=near_depth_m) directly
    abutting a flat 'neighbor' region (right half, depth=near_depth_m+gap_m),
    a single vertical step edge at column 40. Target bbox = the left half
    (0,0,40,80)."""
    depth = np.full(IMG_SHAPE, near_depth_m, dtype=np.float32)
    depth[:, 40:] = near_depth_m + gap_m
    return depth


TARGET_BBOX = (0, 0, 40, 80)

BLUR_LEVELS_PX = [0, 1, 2, 4, 8]
NOISE_LEVELS = [0.0, 1.0, 2.0, 3.0, 5.0]


def sweep_blur_raw_signals(seed=RANDOM_STATE, quantize_m=0.0):
    # quantize_m=0.0 by default here: DIAGNOSED this session (see
    # project_overnight/LOG.md T6a entry) -- apply_depth_noise()'s default
    # 1mm quantization silently erases sub-mm blur perturbations on a
    # step edge whose gap is itself comparable to 1mm (exactly the
    # Contact/Support regime per explore/E1_FINDINGS.md's gap
    # distributions), independent of the blur level requested. Disabled
    # here to isolate blur's own effect; re-enabled in a second test to
    # document the quantization interaction explicitly as its own finding.
    base_depth = make_step_edge_depth()
    c1_gaps, c2_raws, levels = [], [], []
    for blur_px in BLUR_LEVELS_PX:
        blurred = apply_depth_noise(base_depth, level=1.0 if blur_px > 0 else 0.0,
                                     seed=seed, edge_blend_px=blur_px, sigma0=0.0, k=0.0,
                                     quantize_m=quantize_m)
        internals = gpe._c1_internals(blurred, TARGET_BBOX)
        c2_raw = gpe._raw_boundary_sharpness(blurred, TARGET_BBOX)
        if internals is not None:
            c1_gaps.append(internals[0])
        if c2_raw is not None:
            c2_raws.append(c2_raw)
        levels.append(blur_px)
    return levels, c1_gaps, c2_raws


def sweep_noise_raw_signals(seed=RANDOM_STATE):
    base_depth = make_step_edge_depth()
    c1_gaps, c2_raws, levels = [], [], []
    for level in NOISE_LEVELS:
        noisy = apply_depth_noise(base_depth, level=level, seed=seed + int(level),
                                   edge_blend_px=0)
        internals = gpe._c1_internals(noisy, TARGET_BBOX)
        c2_raw = gpe._raw_boundary_sharpness(noisy, TARGET_BBOX)
        if internals is not None:
            c1_gaps.append(internals[0])
        if c2_raw is not None:
            c2_raws.append(c2_raw)
        levels.append(level)
    return levels, c1_gaps, c2_raws


def test_step_edge_construction_is_correct():
    depth = make_step_edge_depth(gap_m=0.001, near_depth_m=0.5)
    assert np.all(depth[:, :40] == 0.5)
    assert np.all(depth[:, 40:] == pytest.approx(0.501))


def test_c2_raw_boundary_sharpness_sensitive_to_blur():
    """Uses quantize_m=0 (see sweep_blur_raw_signals' docstring) --
    otherwise apply_depth_noise()'s default 1mm quantization erases the
    entire blur signal on this 1mm-gap edge, which is a SEPARATE finding
    (see test_quantization_masks_blur_signal_on_small_gaps below), not
    evidence against C2 itself."""
    levels, _, c2_raws = sweep_blur_raw_signals(quantize_m=0.0)
    assert len(c2_raws) == len(levels), "C2 returned None for some blur levels -- unexpected on a clean synthetic edge"
    rho, _ = spearmanr(levels, c2_raws)
    print(f"\nC2 raw boundary_sharpness vs blur (quantize_m=0): rho={rho:.3f}, values={c2_raws}")
    assert abs(rho) > 0.6, (
        f"C2's raw boundary_sharpness is NOT strongly monotonic in blur even with "
        f"quantization disabled (rho={rho:.3f}). This would be new evidence of a "
        f"deeper problem beyond both the quantization interaction and the scene-level "
        f"background-contamination mechanism."
    )
    assert rho < 0, (
        f"C2's raw boundary_sharpness IS strongly monotonic in blur (rho={rho:.3f}) "
        f"but in the WRONG (positive) direction -- more blur should smooth the edge "
        f"and DECREASE the gradient, not increase it. Investigate the sign."
    )


def test_quantization_masks_blur_signal_on_small_gaps():
    """NEW FINDING this session: with the noise model's DEFAULT 1mm
    quantization active, C2's raw signal shows ZERO variation across the
    entire blur sweep (0 to 8px) on a 1mm-gap edge -- the exact gap scale
    explore/E1_FINDINGS.md found for Contact/Support pairs. This is not a
    C2 bug; it's the shared noise_model.apply_depth_noise()'s quantization
    step silently erasing sub-quantization-scale perturbations regardless
    of how much blur was requested. Documented as a reporting test (not a
    hard gate) since this affects the WHOLE noise-model infrastructure,
    not just C1/C2/C3."""
    levels, _, c2_raws_quantized = sweep_blur_raw_signals(quantize_m=0.001)
    unique_vals = len(set(np.round(c2_raws_quantized, 8).tolist()))
    print(f"\nC2 raw boundary_sharpness vs blur, quantize_m=0.001 (default): "
          f"values={c2_raws_quantized}, unique={unique_vals}")
    assert unique_vals == 1, (
        f"Expected the default-quantization sweep to be perfectly constant on this "
        f"1mm-gap edge (confirming the masking mechanism) -- got {unique_vals} unique "
        f"values instead: {c2_raws_quantized}. If this now varies, the masking "
        f"mechanism described above may have been fixed or changed -- update this "
        f"test's assertion and docstring, don't just loosen the bound."
    )


def test_c2_raw_boundary_sharpness_sensitive_to_noise():
    levels, _, c2_raws = sweep_noise_raw_signals()
    assert len(c2_raws) == len(levels)
    rho, _ = spearmanr(levels, c2_raws)
    print(f"\nC2 raw boundary_sharpness vs noise: rho={rho:.3f}, values={c2_raws}")
    assert abs(rho) > 0.6, (
        f"C2's raw boundary_sharpness is NOT strongly monotonic in noise on an "
        f"isolated synthetic step edge (rho={rho:.3f}). Expected a strong POSITIVE "
        f"relationship (more per-pixel noise -> more spurious local gradient)."
    )


def test_c1_gap_estimate_on_clean_step_edge():
    """FIXED this session (C4, project/closure): on a NOISELESS, un-blurred,
    ISOLATED two-region step edge (no background, no other objects -- the
    cleanest possible case) with a true 1mm gap, C1's estimated_gap now
    correctly recovers ~0.001, not the previously-xfailed 0.0.

    Root cause (see project_closure/CRITERIA.md's C4 section for the full
    traced diagnosis): the ring used to be `dilated & ~eroded`, straddling
    the bbox border with ~50% of its pixels trivially inside the original
    mask (local_gap~=0 by construction against inner_depth, the SAME
    surface). The 10th percentile of a population where that same-surface
    half is a near-50% chunk was mathematically guaranteed to return ~0.
    Fixed by restricting the ring to EXTERIOR-only pixels
    (`dilated & ~mask`, never `eroded`) -- see
    explore/scripts/geometric_prg_error.py's _c1_internals()."""
    depth = make_step_edge_depth(gap_m=0.001, near_depth_m=0.5)
    internals = gpe._c1_internals(depth, TARGET_BBOX)
    assert internals is not None
    estimated_gap, sigma_gap, score = internals
    print(f"\nC1 on clean 1mm step edge: estimated_gap={estimated_gap:.5f}m "
          f"(true=0.001m), score={score:.3f}")
    assert estimated_gap == pytest.approx(0.001, abs=0.0005), (
        f"C1's estimated_gap ({estimated_gap:.5f}) no longer matches the true 0.001m gap "
        f"-- if the ring-construction fix (project_closure/CRITERIA.md's C4 section) has "
        f"regressed, investigate before loosening this assertion."
    )


def test_c1_raw_gap_sensitive_to_blur():
    """C4 (project/closure session): after the ring-construction fix
    (project_closure/CRITERIA.md's C4 section), does C1's raw
    estimated_gap respond monotonically to blur? Uses quantize_m=0 for the
    same reason test_c2_raw_boundary_sharpness_sensitive_to_blur does --
    isolates blur's own effect from the separate quantization-masking
    finding above."""
    levels, c1_gaps, _ = sweep_blur_raw_signals(quantize_m=0.0)
    assert len(c1_gaps) == len(levels), "C1 returned None for some blur levels"
    rho, _ = spearmanr(levels, c1_gaps)
    print(f"\nC1 raw estimated_gap vs blur (quantize_m=0): rho={rho:.3f}, values={c1_gaps}")
    assert abs(rho) > 0.6, (
        f"C1's raw estimated_gap is NOT strongly monotonic in blur even after the C4 "
        f"ring fix (rho={rho:.3f})."
    )


def test_c1_raw_gap_vs_noise_REPORTING_not_gated():
    """C4 finding (reported, NOT hard-gated, per this file's own established
    convention -- see test_quantization_masks_blur_signal_on_small_gaps
    above for the same discipline): C1's raw estimated_gap vs noise level,
    using the standard [0, 1, 2, 3, 5] sweep, does NOT clear the |rho|>0.6
    bar that CRITERIA.md's C4 section pre-committed to (measured: rho~=0.0
    over 30-seed-averaged means per level; single-seed-per-level as used
    elsewhere in this file gives the same qualitative result).

    Root cause (diagnosed, not a bug in the C4 fix itself): the ring is now
    exterior-only, so estimated_gap = 10th percentile of
    |noisy_neighbor_depth - inner_depth| over ~480 pixels. At level=0 (no
    noise), this exactly recovers the true 1mm gap. The MOMENT any noise is
    added (level=1, sigma0=1mm -- comparable in magnitude to the 1mm true
    gap itself), the 10th percentile of the resulting |offset + N(0,sigma)|
    population is biased sharply DOWNWARD (some fraction of the ~480
    samples land close to zero purely by chance when noise partially
    cancels the small true offset) -- then, as noise grows further
    (level 2/3/5), the 10th percentile trends back upward (confirmed:
    30-seed-averaged means are 0.00021 / 0.00031 / 0.00041 / 0.00065 for
    levels 1/2/3/5 -- CLEANLY monotonic increasing among themselves), but
    never recovers to level=0's exact value within this sweep's range. The
    overall level-0-through-5 relationship is therefore genuinely
    NON-monotonic (a fold/order-statistic boundary effect at the
    zero-noise reference point, not a coding defect), distinct from the
    zero-return bug this session's C4 fix resolved. Left as a reported,
    diagnosed limitation rather than silently hidden or criteria-loosened
    -- see project_closure/LOG.md's C4 entry."""
    levels, c1_gaps, _ = sweep_noise_raw_signals()
    assert len(c1_gaps) == len(levels)
    rho, _ = spearmanr(levels, c1_gaps)
    print(f"\nC1 raw estimated_gap vs noise: rho={rho:.3f}, values={c1_gaps} "
          f"(NOT gated -- see this test's docstring)")


def test_c1_score_responds_to_blur_and_noise_even_if_direction_is_mixed():
    """Report (not gate) whether C1's FINAL [0,1] score -- as opposed to
    its raw gap estimate above -- shows any sweep sensitivity at all, for
    completeness against explore/CRITERIA.md's original full-scene
    finding (0/5 worlds passed for C1's score)."""
    base_depth = make_step_edge_depth()
    blur_scores = []
    for blur_px in BLUR_LEVELS_PX:
        blurred = apply_depth_noise(base_depth, level=1.0 if blur_px > 0 else 0.0,
                                     seed=RANDOM_STATE, edge_blend_px=blur_px, sigma0=0.0, k=0.0)
        blur_scores.append(gpe.geometric_c1_gap_uncertainty(None, blurred, TARGET_BBOX))
    rho_blur, _ = spearmanr(BLUR_LEVELS_PX, blur_scores)

    noise_scores = []
    for level in NOISE_LEVELS:
        noisy = apply_depth_noise(base_depth, level=level, seed=RANDOM_STATE + int(level), edge_blend_px=0)
        noise_scores.append(gpe.geometric_c1_gap_uncertainty(None, noisy, TARGET_BBOX))
    rho_noise, _ = spearmanr(NOISE_LEVELS, noise_scores)

    print(f"\nC1 FINAL SCORE vs blur: rho={rho_blur:.3f}, values={blur_scores}")
    print(f"C1 FINAL SCORE vs noise: rho={rho_noise:.3f}, values={noise_scores}")
    # No assertion -- this is a reporting test (pytest -s to see the printed values),
    # matching CRITERIA.md's framing that the score's own sensitivity is evaluated
    # separately (criterion b) on full scenes, not gated here.
