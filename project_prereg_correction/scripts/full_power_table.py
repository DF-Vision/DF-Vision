#!/usr/bin/env python3
"""
Full power table under BOTH the CURRENT (post-0d74cd4, decoupled noise)
and ISOLATED-FIX (null bug fixed, old multiplier-noise coupling
preserved) generators, at N=90 (DEFAULT_N_PER_OBJECT), 120, 180, 195,
each at 1x/2x/3x gap_signal_multiplier. Reduced scale (n_reps=100),
consistent with project_audit/C6_AUDIT.md's methodology. Read-only:
imports the committed mock_data_generator.py unmodified for the CURRENT
column; patches an in-memory copy (never written to disk) for the
ISOLATED column, exactly as in run_isolated_check.py.
"""
import sys
import types
from pathlib import Path

import numpy as np
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = PROJECT_ROOT / "perception-calibration-gap" / "mock-pipeline" / "mock_data_generator.py"
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "mock-pipeline"))

import mock_data_generator as current_gen  # CURRENT, committed, unmodified

CURRENT_LINE = "            noise_scale = OBJECT_DETECTOR_NOISE[obj]\n"
ISOLATED_LINE = (
    "            noise_scale = (OBJECT_DETECTOR_NOISE[obj] * gap_signal_multiplier\n"
    "                           if gap_signal_multiplier > 0 else OBJECT_DETECTOR_NOISE[obj])\n"
)
source = GEN_PATH.read_text()
assert source.count(CURRENT_LINE) == 1
isolated_mod = types.ModuleType("mock_data_generator_isolated")
isolated_mod.__file__ = str(GEN_PATH) + " (ISOLATED, in-memory patch only)"
exec(compile(source.replace(CURRENT_LINE, ISOLATED_LINE), str(GEN_PATH) + " [ISOLATED]", "exec"),
     isolated_mod.__dict__)

MODEL_B_FORMULA = "failure ~ true_occlusion + Gap_initial + C(object_id)"
N_REPS = 100
BASE_SEED = 20260921


def empirical_power(gen_module, n_per_object, mult, n_reps, base_seed):
    hits, converged = 0, 0
    for rep in range(n_reps):
        seed = base_seed * 100_000 + rep
        df = gen_module.generate_mock_trials(n_per_object=n_per_object, seed=seed, gap_signal_multiplier=mult)
        if df["failure"].nunique() < 2:
            continue
        try:
            m = smf.logit(MODEL_B_FORMULA, data=df).fit(disp=0, maxiter=100)
            pval = m.pvalues.get("Gap_initial", np.nan)
            coef = m.params.get("Gap_initial", np.nan)
            if np.isnan(pval):
                continue
            converged += 1
            if pval < 0.05 and coef > 0:
                hits += 1
        except Exception:
            continue
    return hits / n_reps


N_CONFIGS = {
    90: {"apple": 30, "purse": 30, "glass": 30},
    120: {"apple": 40, "purse": 40, "glass": 40},
    180: {"apple": 60, "purse": 60, "glass": 60},
    195: {"apple": 65, "purse": 65, "glass": 65},
}

print(f"n_reps={N_REPS} per cell, base_seed={BASE_SEED} (reduced scale, per audit methodology)\n")
print(f"{'N':>5} {'mult':>5} | {'CURRENT (decoupled)':>20} | {'ISOLATED (null-fixed, coupled)':>30}")
print("-" * 70)
for n_total, n_per_object in N_CONFIGS.items():
    for mult in [1.0, 2.0, 3.0]:
        p_current = empirical_power(current_gen, n_per_object, mult, N_REPS, BASE_SEED)
        p_isolated = empirical_power(isolated_mod, n_per_object, mult, N_REPS, BASE_SEED)
        print(f"{n_total:>5} {mult:>5.1f} | {p_current:>20.3f} | {p_isolated:>30.3f}")
