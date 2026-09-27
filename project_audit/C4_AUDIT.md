# C4 Audit — the "unchanged, slightly worse on one metric" verdict

Branch `project/audit`, from `project/closure`. All re-runs done from a
scratch clone at `/tmp/audit_scratch_c4` (never touching the committed
repo) or via in-memory functions; nothing in `perception-calibration-gap/`,
`explore/`, `.joblib`, or `integration/prg_config.json` was modified.

## 1. Exact before/after numbers, quoted

**Pre-fix** — `explore/e3_output_SIM.txt` (committed file, predates the
`project/closure` session), criterion (c):
```
n=262, wrong fraction=0.737
C1: AUROC=0.4567, CI=[0.1756,0.5672] -> FAIL
C2: AUROC=0.4857, CI=[0.3453,0.5757] -> FAIL
C3: AUROC=0.4625, CI=[0.1858,0.5750] -> FAIL
```

**Post-fix** — `project_closure/results/c4_e3_rerun_SIM_output.txt`
(committed by the closure session), criterion (c):
```
n=262, wrong fraction=0.821
C1: AUROC=0.2796, CI=[0.0802,0.3971] -> FAIL
C2: AUROC=0.3896, CI=[0.2771,0.5123] -> FAIL
C3: AUROC=0.2855, CI=[0.0863,0.3929] -> FAIL
```
Criterion (a) (non-constant/range/deterministic): PASS, unchanged, both
runs. Criterion (b) (sensitivity, rho>0.6 on ≥4/5 worlds): FAIL, 0/5
worlds, unchanged, both runs, per-world rho values identical between the
two committed files to 3 decimal places (verified: `explore/e3_output_SIM.txt`
and `c4_e3_rerun_SIM_output.txt`'s (b) sections are byte-identical). This
makes sense — the fix only touches `_c1_internals`'s ring definition,
which criterion (b)'s per-world rho values (computed the same way in both
runs) evidently don't depend on in a way that shows up in this specific
metric; C4's own re-run correctly reports this as "unchanged," which is
accurate.

## 2. CI-overlap verdict — literal answer, then a stricter check

**Literal question asked**: is the post-fix value inside or outside the
pre-fix value's CI?

- C1: post-fix point estimate (0.2796) **is inside** the pre-fix CI
  [0.1756, 0.5672].
- C1 reverse check: pre-fix point estimate (0.4567) **is outside** the
  post-fix CI [0.0802, 0.3971] (0.4567 > 0.3971).

This is an asymmetric, ambiguous result by itself — checking whether one
point estimate falls in the other's *marginal* CI is not a rigorous test
of whether two estimates differ (marginal CIs are wider than a CI on
their difference would be, precisely because they don't exploit the
correlation between the two measurements on the same underlying rows).
**No CI on the difference itself was computed by the closure session** —
this is exactly the gap this audit was asked to check.

**Stricter check performed for this audit**: a PAIRED, world-grouped
bootstrap (2000 resamples, same resampled world set applied to both the
pre-fix and post-fix estimator in every resample) on AUROC(post) −
AUROC(post), using the exact same 262 rows for both estimators. Script:
`/tmp/audit_scratch_c4/scratch_audit/paired_comparison.py` (not
committed — scratch only, per hard rule 4). Exact output:
```
n=262 rows with BOTH pre-fix and post-fix internals available
pre-fix wrong fraction=0.7366 (committed pre-fix run reported 0.737 on n=262 -- compare)
post-fix wrong fraction=0.8206 (committed post-fix run reported 0.821 on n=262 -- compare)

AUROC pre-fix (this re-derivation)  = 0.4567  (committed: 0.4567)
AUROC post-fix (this re-derivation) = 0.2796  (committed: 0.2796)

Paired world-grouped bootstrap (n_boot_used=2000/2000) on AUROC(post) - AUROC(pre):
  mean diff = -0.1664
  95% CI    = [-0.2221, -0.0928]
  CI excludes 0 (i.e. change is distinguishable from noise): YES
```
**Verdict on this question**: the naive marginal-CI-containment check is
ambiguous (as reported above), but the correct paired test shows the
AUROC degradation for C1 **is** statistically distinguishable from
random noise (95% CI on the difference excludes 0, and is entirely
negative). The closure session's LOG.md characterization ("measurably
worse") holds up under a more rigorous test than it actually performed —
if anything, the closure session's own evidence for "worse" was weaker
than what's actually true.

## 3. Independent reproduction

Ran `explore/scripts/e3_evaluate_candidates.py` from a fresh
`git clone` of the repo at `project/closure`'s HEAD (`/tmp/audit_scratch_c4`,
commit `9c0d39e`), not reusing any cached result:
```
$ cd /tmp/audit_scratch_c4 && python3 explore/scripts/e3_evaluate_candidates.py
(a) ... C1: n=263, range=[0.0000,1.0000], unique=192 ... -> PASS
(b) ... C1: noise rho>0.6 on 0/5 worlds, blend rho>0.6 on 0/5 worlds -> FAIL
(c) ... n=262, wrong fraction=0.821
    C1: AUROC=0.2796, CI=[0.0802,0.3971] -> FAIL
    C2: AUROC=0.3896, CI=[0.2771,0.5123] -> FAIL
    C3: AUROC=0.2855, CI=[0.0863,0.3929] -> FAIL
```
**Bit-for-bit identical** to the committed `c4_e3_rerun_SIM_output.txt`
(the evaluation is fully seeded — `RANDOM_STATE=20260920` — so this is
expected, and confirms the committed result file wasn't hand-edited or
misreported). Reproduction: **CONFIRMED**.

## 4. Out-of-sample sanity check (3 new synthetic scenes, not in `explore/`'s test set)

Script: `/tmp/audit_scratch_c4/scratch_audit/out_of_sample_check.py`
(scratch only). Uses the CURRENT (post-fix) `_c1_internals` directly.

**Scenes 1-3 — object pair at 3 camera distances, true gap = 1cm each:**
```
distance=0.3m: estimated_gap=0.010000m  true_gap=0.01m  raw_error=0.000000m
distance=1.0m: estimated_gap=0.010000m  true_gap=0.01m  raw_error=0.000000m
distance=3.0m: estimated_gap=0.010000m  true_gap=0.01m  raw_error=0.000000m
```
Zero error at all three distances — the fix's ring/percentile logic is
scale-invariant in this idealized (noiseless) case. No new artifact
found here.

**Scene 4 — target partially occluded by a nearer object** (occluder at
depth 0.1m covering roughly 33% of the target bbox's `INSET_PX` interior
sampling region):
```
estimated_gap=0.010000m  true_gap=0.01m  raw_error=0.000000m  score=0.9701
inner_depth median actually used: median=0.5000m (target's true surface is 0.5m -- NOT contaminated)
```
Zero error — the occluder covers less than 50% of the interior sampling
region, and `inner_depth` is computed via `np.median`, which is robust
below that threshold. **This robustness is NOT attributable to the C4
fix** — `inner_depth`'s computation (the `INSET_PX` interior + median)
was not touched by C4's change, which only modified the ring definition.

**Follow-up check (beyond the 3 requested scenes, done to characterize
the boundary of the above finding)**: an occluder covering ~83% of the
interior:
```
large occluder (>50% of interior): (0.4099999964237213, 0.2108185151780677, 0.06432329701185001) true_gap=0.01
```
estimated_gap = 0.41m vs true 0.01m — a raw error of 0.40m, i.e. the
estimate breaks down completely once occluder coverage exceeds ~50% of
the interior sample (median flips to the occluder's depth). **This is a
real, pre-existing limitation, not a new artifact introduced by C4** —
it exists in the un-touched `inner_depth` code path and would have been
present before the fix too (verified by inspection: the pre-fix
`_c1_internals` used the identical `inner`/`INSET_PX`/median logic,
only the ring differed).

**Conclusion for this task**: no new failure mode was found that the
fix introduced. The one real limitation found (interior contamination
above ~50% occluder coverage) predates C4 and is orthogonal to it.

## 5. Plain-language verdict

**The PRG negative result holds up, and the audit found it is actually
better-supported than the closure session's own reporting showed.** The
fix is real and reproducible (confirmed independently, bit-for-bit). The
"unchanged" claim for criteria (a) and (b) is accurate and directly
verified. The "measurably worse" claim for criterion (c) was reported
with only a naive, ambiguous marginal-CI comparison by the closure
session — this audit ran the correct paired test and found the
degradation on C1's AUROC IS statistically distinguishable from noise
(95% CI on the difference: [-0.222, -0.093], excluding zero), which is
stronger evidence than the closure session presented. Separately, this
audit found no new artifact introduced by the fix itself when tested on
three out-of-sample synthetic scenes (varying camera distance, and a
partial occluder) — the fix behaves exactly as documented in those
cases, and the one real limitation surfaced (median-based interior
sampling breaking down above ~50% occlusion) is a pre-existing property
of code the fix did not touch, not a regression.
