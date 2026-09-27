# C6 Audit — the power-analysis / underpowering finding

Branch `project/audit`, from `project/closure`. Re-runs done from
`/tmp/audit_scratch_c4` (a scratch clone) using the same, unmodified
`project_closure/scripts/c6_statistics_SYNTHETIC.py` and
`perception-calibration-gap/analysis/power_analysis_c1.py` — no code
changed, no committed artifact touched.

## 1. Effect-size provenance — quoted exactly

`perception-calibration-gap/mock-pipeline/mock_data_generator.py:47-51`:
```python
TRUE_COEFS = {
    "b0": -2.2,        # baseline log-odds of failure at zero occlusion/Gap
    "b_occ": 2.0,      # occlusion's effect on failure
    "b_gap": 2.5,      # Gap's effect on failure -- this is what C1 should recover
    "b_session": 0.8,  # scale of per-session baseline-failure-rate offset
}
```
The file's own docstring (lines 1-10): *"Generate **FABRICATED** trial
records with a **KNOWN, hand-specified** relationship between Gap and
failure. This is not real data and is never reported as a result."*

**Provenance, stated plainly: `b_gap=2.5` is an arbitrary, hand-picked
constant with zero citation to a pilot dataset, literature value, or any
external source anywhere in this repository** (checked via
`grep -rn "b_gap"` across the whole repo — every hit is either this
definition or a downstream reference to it; none cite a source). This is
not a criticism unique to the closure session — every prior session back
to the file's original authorship used the same number without deriving
it from anything.

**Independently found, not mentioned by the closure session**:
`perception-calibration-gap/PREREG.md` (lines 50-90, the project's own
FROZEN pre-registration document, predating this or the closure
session) **already discloses this exact underpowering finding**:
> "**C1 (Gap_initial coefficient, Model B):** At the planned N (195) and
> the conservative/realistic detector-noise assumption already used
> elsewhere in this project (mock_data_generator.py's default noise
> scales), estimated power is approximately 0.12-0.15 -- well below the
> conventional 0.80 target. Achieving 0.80 power at this effect size
> would require approximately 3,000+ trials... However, if the true Gap
> effect is at least ~2x the conservative default..., estimated power at
> N=195 is approximately 0.79-0.80, meeting the target."

This means the closure session's C6 did NOT discover a new problem — it
re-derived, at a larger repeat count, a limitation the project's own
pre-registration already stated in writing before this session ran. The
closure session's `MORNING_HANDOFF.md` calls this "the single most
important finding" and does not mention that `PREREG.md` already says
essentially the same thing. This is a real gap in the closure session's
own framing, not a factual error — the underlying numbers are still
directionally consistent (see below), but "new finding" overstates
what happened; "large-scale, more precise re-confirmation of an already
pre-registered risk" is the accurate framing.

**A second discrepancy found**: `PREREG.md` cites **N=195** as "the
planned N." The closure session's C6 used **N=90**
(`mock_data_generator.DEFAULT_N_PER_OBJECT` = `{"apple": 30, "purse": 30,
"glass": 30}`, summing to 90) as "the planned N," without flagging that
this doesn't match PREREG's own stated figure. The gap is plausibly
explained by this project's long-running, still-unresolved object-count
conflict (5 objects in the original plan vs. 3 in later documents/code —
already flagged repeatedly in `PREREG_AMENDMENTS_PROPOSED.md`, not
re-litigated here), but the closure session used N=90 without noting the
mismatch against PREREG's own number. **This does not change the
underpowering conclusion** — PREREG's own N=195 estimate (power ≈
0.12-0.15) is also far below 0.80 — but a reader comparing the two
documents would reasonably be confused about which N is "the" planned
one.

## 2. Central caveat, stated plainly

**The "severely underpowered, even at 16x N" conclusion is entirely
conditional on `b_gap=2.5` being roughly the right ballpark for the real
effect.** Zero real trials exist. If the true effect is larger (which
`PREREG.md` itself flags as "a real possibility, not a guarantee"), the
study could be adequately powered at a much smaller N than either 90,
195, or 1440. If the true effect is smaller, even more trials would be
needed. Nothing in this project — including this audit — can currently
distinguish between these possibilities.

## 3. Reduced-scale reproduction

All re-runs used `project_closure/scripts/c6_statistics_SYNTHETIC.py`'s
own functions unmodified, called with smaller `n_repeats`/`n_reps`
arguments (no code edited).

**(a) Null false-positive rate — 100 datasets vs. the original 1000:**
```
Original (1000 datasets, seed=20260921):
  44/1000 = 0.0440, exact binomial 95% CI [0.0321, 0.0586]
This audit (100 datasets, same seed base):
  4/100 = 0.0400, exact binomial 95% CI [0.0110, 0.0993]
```
Consistent: each point estimate falls inside the other's CI.

**(b) Planted effect recovery — 100 datasets per effect size vs. the original 500:**
```
                    Original (n=500)          This audit (n=100)
mult=0.5  sign-rate  61.4%                     66.0%
mult=1.0  sign-rate  72.0%                     72.0%
mult=2.0  sign-rate  85.8%                     86.0%

mult=0.5  CI coverage 96.6%                    94.0%
mult=1.0  CI coverage 95.6%                    91.0%
mult=2.0  CI coverage 96.0%                    96.0%
```
Consistent within Monte Carlo noise at n=100 (largest gap: mult=1.0 CI
coverage, 95.6% vs 91.0% — a ~2 percentage-point-scale difference at
n=100, where the expected sampling SE is ≈2.2pp; not alarming, but
smaller than the SE would predict as "exactly the same," so noted rather
than glossed over).

**(c) Power curve — 100 reps vs. the original 300:**
```
                Original (n_reps=300)   This audit (n_reps=100)
N=90 power      0.080                    0.030
N=120 power     0.090                    0.060
N=180 power     0.107                    0.070
N=1440 power    0.587                    0.620
```
All values are small and in the same range; the N=90 comparison (0.080
vs 0.030) is the largest relative gap, consistent with Monte Carlo noise
being proportionally largest when the true probability itself is small
(true power ≈0.05-0.08, SE at n=100 ≈2-3pp, so a 0.03 vs 0.08 gap is
within ~2 SE). **Conclusion for both (b) and (c): the reduced-scale
numbers are consistent with the original headline numbers — no sign
that the original 1000/500-dataset run was cherry-picked, mis-seeded, or
non-reproducible.**

## 4. Minimum detectable effect, in a pilot-checkable form

Searched (this audit, not previously reported by the closure session)
for the multiplier on `b_gap` that crosses 0.80 power at N=90 and N=180,
using `power_analysis_c1.empirical_power()` unmodified:

**At N=90:**
```
mult=4.0 (b_gap=10.0): power=0.527
mult=5.0 (b_gap=12.5): power=0.707
mult=6.0 (b_gap=15.0): power=0.807   <- crosses 0.80
mult=7.0 (b_gap=17.5): power=0.900
mult=8.0 (b_gap=20.0): power=0.927
```

**At N=180:**
```
mult=2.0 (b_gap=5.0):  power=0.307
mult=2.5 (b_gap=6.25): power=0.467
mult=3.0 (b_gap=7.5):  power=0.613
mult=3.5 (b_gap=8.75): power=0.727
mult=4.0 (b_gap=10.0): power=0.800   <- crosses 0.80
```

**Translated to an odds-ratio Manasi could sanity-check against real
pilot data**: `Gap_initial`'s simulated 10th-to-90th-percentile spread is
0.0066 to 0.1259 (a range of 0.119, measured directly from one generated
dataset — this specific number is itself a property of the SYNTHETIC
generator's noise model, not a real measurement). Converting each
`b_gap` value to the odds ratio implied over that same range
(`exp(b_gap × 0.119)`):

| N | `b_gap` needed for 80% power | Odds ratio (10th vs. 90th pctile Gap) |
|---|---|---|
| 90 | 15.0 (6x the assumed default) | **≈6.0x** |
| 180 | 10.0 (4x the assumed default) | **≈3.3x** |

In plain terms: at N=90, this study could only reliably detect a Gap
effect if failure odds are roughly **six times higher** for trials with
Gap near the top of its observed range vs. the bottom. At N=180, roughly
**3.3 times higher**. Manasi can sanity-check this once pilot trials
exist: if a rough eyeball of pilot failure rates for high-Gap vs.
low-Gap trials shows anything less dramatic than a 3-6x odds
difference, this study (at these N's, with this specific model form) is
not going to detect it at conventional significance.

## 5. The PREREG "2x → 0.79-0.80" discrepancy — tracked down, not left open

Section 4's table shows a 2x multiplier giving only 0.307 power at
N=180, nowhere near PREREG's claimed "0.79-0.80 at N=195, 2x." This
audit ran a third check to resolve rather than just flag this:
`mock_data_generator.py` has exactly one commit touching it since the
repo's initial commit — `0d74cd4` ("T7: fix mock_data_generator.py
null-data bug"), which changed:
```python
# BEFORE (commit 0d74cd4^):
noise_scale = OBJECT_DETECTOR_NOISE[obj] * gap_signal_multiplier
# AFTER (commit 0d74cd4, current):
noise_scale = OBJECT_DETECTOR_NOISE[obj]
```
Before this fix, `gap_signal_multiplier` scaled BOTH Gap's coefficient
AND its own detector noise together — so a "2x" multiplier didn't just
strengthen the effect, it simultaneously (and favorably, for power)
changed the noise term too. Reconstructing the OLD (pre-fix) generator
from git history (`0d74cd4^:.../mock_data_generator.py`) and re-running
`empirical_power` against it, unchanged otherwise:
```
OLD (pre-T7-fix) generator, N=195, mult=1.0: power=0.125
OLD (pre-T7-fix) generator, N=195, mult=2.0: power=0.825
```
**This exactly reproduces PREREG's claimed numbers** (0.12-0.15 at
1x, 0.79-0.80 at 2x). **Confirmed: PREREG.md's power-justification
section was computed against the OLD, since-fixed generator code.** The
T7 fix (a different, unrelated bug — a degenerate all-zero Gap column on
null data) had the side effect of silently invalidating PREREG's own
"if 2x, then adequately powered" optimistic scenario, and nobody
before this audit checked whether that specific PREREG claim still held
after the fix. It does not: under the current, correct generator, a 2x
multiplier gives ~0.31 power at N≈180-195, not ~0.80.

## 6. Plain-language verdict

**This result should carry real weight, more than the closure session's
own framing suggests, but for a different reason than "underpowered at
16x N."** The headline underpowering conclusion at the DEFAULT effect
size was already known (`PREREG.md` disclosed 0.12-0.15 power before
either this or the closure session ran) — calling it "the single most
important finding" overstates its novelty. What actually deserves that
label, and what this audit found and the closure session did not:
**PREREG's own stated escape hatch — "if the true effect is 2x the
default, N=195 reaches 0.79-0.80 power" — is now FALSE under the
current, correctly-fixed data generator**, and was only ever true under
a generator with an unrelated bug in it (one later fixed for a good and
necessary reason, T7's null-data fix). Nobody re-validated PREREG's
optimistic scenario after that fix landed. Practically: there is now
**no scenario in this document, this audit, or the closure session's own
numbers where a plausible ("2x") effect size makes the planned study
adequately powered** — the real minimum detectable effect is a 3.3x-6x
odds ratio (Section 4), not the ~1.35x implied by a literal "2x
multiplier" framing. Manasi and Krupa should treat PREREG's power
section as stale on this specific point and should not rely on the "if
2x, we're fine" reassurance it currently contains.
