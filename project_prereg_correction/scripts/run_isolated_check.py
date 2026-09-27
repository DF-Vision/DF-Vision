#!/usr/bin/env python3
"""
Loads the CURRENT (post-0d74cd4) mock_data_generator.py source, patches
ONLY the noise_scale line to the "isolated fix" variant (fixes the
null-multiplier degenerate case, preserves the OLD multiplicative
coupling at every other multiplier value), execs it as a fresh module,
and runs the power simulation under it. Never writes to the committed
file -- reads its text, patches an in-memory copy, execs that.
"""
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = PROJECT_ROOT / "perception-calibration-gap" / "mock-pipeline" / "mock_data_generator.py"

CURRENT_LINE = "            noise_scale = OBJECT_DETECTOR_NOISE[obj]\n"
ISOLATED_LINE = (
    "            noise_scale = (OBJECT_DETECTOR_NOISE[obj] * gap_signal_multiplier\n"
    "                           if gap_signal_multiplier > 0 else OBJECT_DETECTOR_NOISE[obj])\n"
)

source = GEN_PATH.read_text()
count = source.count(CURRENT_LINE)
assert count == 1, f"expected exactly 1 occurrence of the current noise_scale line, found {count}"
patched_source = source.replace(CURRENT_LINE, ISOLATED_LINE)
assert patched_source != source

mod = types.ModuleType("mock_data_generator_isolated")
mod.__file__ = str(GEN_PATH) + " (ISOLATED-FIX in-memory patch, not written to disk)"
exec(compile(patched_source, str(GEN_PATH) + " [ISOLATED]", "exec"), mod.__dict__)
sys.modules["mock_data_generator_isolated"] = mod

# Sanity: confirm the null case no longer produces a degenerate zero-variance column.
df_null = mod.generate_mock_trials(seed=1, gap_signal_multiplier=0.0)
print(f"Sanity (null case, mult=0.0): Gap_initial variance = {df_null['Gap_initial'].var():.6f} "
      f"(must be > 0 -- confirms the null-data bug IS fixed under the isolated variant)")

# Sanity: confirm multiplier=2.0 noise_scale matches the OLD (pre-0d74cd4) coupled
# behavior, not the current decoupled one, by comparing Gap_initial's variance at
# mult=1.0 vs mult=2.0 -- under the OLD coupling, variance should visibly increase
# with the multiplier (more detector noise); under the CURRENT decoupled code,
# variance is identical between mult=1.0 and mult=2.0 at the SAME seed's noise draws
# (only the coefficient / TRUE_COEFS scaling changes, not the noise itself).
df_m1 = mod.generate_mock_trials(seed=2, gap_signal_multiplier=1.0)
df_m2 = mod.generate_mock_trials(seed=2, gap_signal_multiplier=2.0)
print(f"Sanity (coupling check): Gap_initial var at mult=1.0 = {df_m1['Gap_initial'].var():.6f}, "
      f"at mult=2.0 = {df_m2['Gap_initial'].var():.6f} "
      f"(OLD/isolated behavior: var should be visibly LARGER at mult=2.0 since noise_scale "
      f"doubles too; CURRENT/decoupled behavior would keep these closer)")

sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "mock-pipeline"))
import numpy as np
import statsmodels.formula.api as smf

MODEL_B_FORMULA = "failure ~ true_occlusion + Gap_initial + C(object_id)"


def empirical_power_isolated(n_per_object, mult, n_reps, base_seed):
    hits, converged = 0, 0
    for rep in range(n_reps):
        seed = base_seed * 100_000 + rep
        df = mod.generate_mock_trials(n_per_object=n_per_object, seed=seed, gap_signal_multiplier=mult)
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
    return hits / n_reps, converged / n_reps


print("\n=== Isolated-fix power simulation (null bug fixed, old multiplier-coupling preserved) ===")
n_per_object_195 = {"apple": 65, "purse": 65, "glass": 65}
for mult in [1.0, 2.0, 3.0]:
    power, conv = empirical_power_isolated(n_per_object_195, mult, 150, 777)
    print(f"N=195, mult={mult} (isolated fix): power={power:.3f}  converged_frac={conv:.3f}")
