# Diagnosis of commit `0d74cd4`

Branch `project/prereg-correction`, from `project/audit`. All re-runs
done from a scratch clone (`/tmp/prereg_scratch`, never committed) or
in-memory source patching (a modified copy of `mock_data_generator.py`'s
text is `exec()`'d into a fresh module object — the committed file on
disk is never written to). No `.joblib`, dataset, or config default
touched; `PREREG.md`/`SCHEMA.md `/`INTERFACE.md ` not edited.

## 1. Full commit, quoted exactly

```
$ git show 0d74cd4 --stat
commit 0d74cd4de6e72465335053195e70ed6a026212d1
Author: Manasi <manasimvs11@gmail.com>
Date:   Mon Sep 21 00:50:38 2026 +0530

    T7: fix mock_data_generator.py null-data bug (gap_signal_multiplier wrongly zeroed detector noise, not just Gap's coefficient) + synthetic recovery check

    Found via T7's pre-committed synthetic-recovery check: gap_signal_multiplier=0.0
    (the documented 'null' setting) also zeroed noise_scale, making
    Gap_initial IDENTICALLY 0.0 for every row -- a constant column causing
    smf.logit() to hit a singular Hessian 100/100 times on null data. Fixed
    by decoupling noise_scale (always full detector noise) from
    gap_signal_multiplier (still scales ONLY the Gap coefficient in the
    failure logit, matching the documented intent). After the fix: null-data
    false-positive rate is 5/100 (5.0%), essentially exact nominal-alpha
    calibration.

    Planted-effect recovery at realistic N (~90 trials): 30/50 (60%) correct
    sign -- FAILS the pre-committed 90% bar, but consistent with and
    empirically confirms PREREG.md's own already-disclosed power analysis
    (12-15% power at the larger N=195 in the original plan). Conformal
    coverage on planted data: 0.955 vs 0.900 target, 5.5pp outside the
    pre-committed +-5pp tolerance (single-seed check, not repeated).

 .../mock-pipeline/mock_data_generator.py           |  22 +++-
 .../scripts/t7_synthetic_recovery_check.py         | 129 +++++++++++++++++++++
 2 files changed, 149 insertions(+), 2 deletions(-)
```

**The exact code change** (the only functional line changed; the rest of
the diff is a comment block and a new, unrelated verification script):
```diff
-            noise_scale = OBJECT_DETECTOR_NOISE[obj] * gap_signal_multiplier
+            noise_scale = OBJECT_DETECTOR_NOISE[obj]
```

## 2. Quantitative parameter change

`noise_scale` is the standard deviation of the Gaussian noise applied to
`perceived_visibility_initial` (a value in [0,1]) when reading
`true_visibility` — NOT a depth measurement. `OBJECT_DETECTOR_NOISE =
{"apple": 0.03, "purse": 0.07, "glass": 0.12}` (visibility-fraction
units, unitless).

| `gap_signal_multiplier` | OLD `noise_scale` | NEW `noise_scale` | Changed? |
|---|---|---|---|
| 0.0 (null) | `base × 0.0 = 0` (degenerate) | `base` (full noise) | YES — this is the bug fix |
| 1.0 (nominal default) | `base × 1.0 = base` | `base` | **NO — identical** |
| 2.0 ("optimistic", used in PREREG's fallback scenario) | `base × 2.0` (doubled) | `base` (unchanged) | YES |
| 3.0+ | `base × mult` (scales linearly) | `base` (constant) | YES |

Confirmed directly (this audit, not assumed):
```
Sanity (null case, mult=0.0), CURRENT code: Gap_initial variance = 0.0 exactly (before fix)
Sanity (null case, mult=0.0), reconstructed ISOLATED-fix code: Gap_initial variance = 0.002600 (fix confirmed)
```

## 3. Commit's own stated purpose

The commit message's ENTIRE stated rationale is about the `mult=0.0`
null case (see §1's full quote) — it never discusses `mult>1` behavior
or acknowledges that the same line change also alters power-analysis
scenarios at `mult=2.0`/`3.0`. The phrase "matching the documented
intent" is the commit's only claim about non-null multiplier values, and
it does not cite a specific document.

**Independently found**: `perception-calibration-gap/analysis/power_analysis_c1.py`
— the file that actually runs the power analysis PREREG.md's fallback
scenario is based on — has this docstring, **dated to the repo's initial
commit (`f1065c2`, 2026-07-30), unmodified since, ~7 weeks before
`0d74cd4` (2026-09-21)**:
```
Two things are varied independently:
    1. gap_signal_multiplier -- how strong the real Gap->failure effect is
       (1.0 = the "realistic-magnitude" default already used elsewhere in
       the project; >1.0 = optimistic scenarios in case the real effect
       turns out stronger than the conservative default).
    2. scale_factor -- how many trials per object...
```
This is genuine, dated, pre-existing documentation (verified via
`git log --format="%h %ad %s" -- perception-calibration-gap/analysis/power_analysis_c1.py`,
one commit, the initial one) — not something written after the fact to
justify `0d74cd4`. It explicitly states `gap_signal_multiplier` should
vary effect strength **independently** of everything else. The pre-fix
coupling (`noise_scale` also scaling with the same multiplier) directly
contradicts this file's own, older, stated design.

**Verdict for this step**: the noise-model change was **(b) an
incidental side effect of fixing the null-case bug, but one that
happened to also correct a separate, pre-existing, and separately
documented design inconsistency** — not (a) explicitly the stated point
of the commit (the commit message never mentions it), and not (c)
undocumented (a dated, independent source does document the intended
semantics, even though the commit didn't cite it).

## 4. Real-world noise-magnitude reference check — UNRESOLVABLE as asked

The repo's only documented, literature-calibrated noise model is
`retrain/scripts/noise_model.py`'s Nguyen, Izadi & Lovell (2012)
Kinect-v1 axial noise model:
```
sigma(z) = sigma0 + k * (z - z_ref)^2   [metres]
sigma0=0.001, k=0.0035, z_ref=0.5
```
This calibrates **raw depth-sensor noise in METERS**, used in a
different subsystem entirely (`explore/scripts/`'s ray-cast PRG-candidate
evaluation, operating on synthetic depth IMAGES).

`OBJECT_DETECTOR_NOISE` (0.03/0.07/0.12) is noise on
`perceived_visibility_initial`, a **unitless VISIBILITY FRACTION in
[0,1]**, used in `mock_data_generator.py`'s completely separate Gap-only
statistics pipeline. There is no depth-to-visibility-fraction conversion
documented anywhere in this repo that would let these two numbers be
compared on a common scale.

**This audit cannot determine whether the pre- or post-`0d74cd4` noise
magnitude is "closer to realistic" using the comparison the task
requested, because no applicable real-world reference for THIS
specific noise term exists in the repository.** Stating this plainly
per hard rule 6, rather than forcing an answer by comparing incompatible
units.

## 5. Isolated-fix test — are the two changes actually separable?

Reconstructed a THIRD variant that fixes ONLY the `mult=0.0` degenerate
case while preserving the OLD multiplicative coupling at every other
multiplier value:
```python
noise_scale = (OBJECT_DETECTOR_NOISE[obj] * gap_signal_multiplier
               if gap_signal_multiplier > 0 else OBJECT_DETECTOR_NOISE[obj])
```
(`project_prereg_correction/scripts/run_isolated_check.py`, exec'd from
an in-memory-patched copy of the current file's source — the committed
file was never modified.)

```
Sanity (null case, mult=0.0): Gap_initial variance = 0.002600 (> 0 -- null bug IS fixed)
Sanity (coupling check): Gap_initial var at mult=1.0 = 0.002580, at mult=2.0 = 0.008425
  (variance visibly grows with multiplier -- OLD coupling preserved, confirmed)

N=195, mult=1.0 (isolated fix): power=0.093
N=195, mult=2.0 (isolated fix): power=0.800
N=195, mult=3.0 (isolated fix): power=1.000
```
**Confirmed: the two changes bundled in `0d74cd4` ARE separable.** A
fix that addresses ONLY the crash-causing null-data bug, without
touching behavior at `mult>0`, is straightforward to construct and
reproduces PREREG's stated "2x → ~0.80" fallback almost exactly (0.800
vs. PREREG's stated 0.79-0.80). The noise-decoupling for `mult>0` was
therefore not *required* to fix the reported bug.

**However**, this does not mean the isolated/old coupling is the
"right" one to use going forward — see §6.

## 6. Why the isolated/old-coupling power numbers are themselves misleading

Mechanistic explanation, verified directly above: under the OLD/isolated
coupling, `gap_signal_multiplier` simultaneously (a) scales up the
failure-logit coefficient (`TRUE_COEFS["b_gap"] * gap_signal_multiplier`)
AND (b) scales up `Gap_initial`'s own variance/spread (since
`noise_scale` feeds directly into how noisy — and therefore how
widely-spread — `Gap_initial` is). **Both effects independently increase
statistical power**: a larger coefficient makes each unit of Gap matter
more for the outcome; a wider Gap_initial spread gives the regression
more predictor variation to estimate that coefficient precisely.

This means the OLD/isolated "2x" scenario does NOT represent "the true
effect is twice as strong, everything else about the measurement stays
the same" — it represents "the true effect is twice as strong AND Gap is
measured with twice the noise-driven spread," a compound, more optimistic
scenario that `power_analysis_c1.py`'s own dated docstring's "varied
independently" framing explicitly did not intend.

## 7. Verdict for this diagnosis

**Load-bearing / incidental / entangled — final classification: the two
changes are demonstrably SEPARABLE (not entangled/inseparable), and the
noise-decoupling half, while not required to fix the stated bug, is
methodologically CORRECT relative to this project's own dated, pre-existing
documentation of what `gap_signal_multiplier` is supposed to represent.**
The realism-reference comparison the task also asked for could not be
answered (§4) — no comparable real-world noise reference exists for this
specific term in this repo. The isolated-fix experiment (§5) reproduces
PREREG's stated fallback number closely, but §6 shows that reproduction
comes from a conflated, double-counted optimism (bigger effect + bigger
predictor spread, applied together), not from a "more realistic" noise
level — so recovering PREREG's number this way does not validate it,
it explains where it came from.
