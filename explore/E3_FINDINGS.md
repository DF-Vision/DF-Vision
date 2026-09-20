# E3 — Candidate Evaluation Findings

Raw output: `explore/e3_output_SIM.txt` (one run, all of criteria a-e on
the 5 non-held-out worlds; `world6_dense_mixed` was **NOT touched** —
see "held-out world" note below).

## Result: decisive negative. No candidate selected.

| Criterion | C1 | C2 | C3 |
|---|---|---|---|
| (a) non-constant/range/deterministic | PASS | PASS | PASS |
| (b) sensitivity (rho>0.6 on ≥4/5 worlds) | **FAIL** (0/5 worlds, noise or blend) | **FAIL** (0/5) | **FAIL** (0/5) |
| (c) validity (AUROC>0.70, CI lower>0.55) | **FAIL** (AUROC=0.457, CI [0.176,0.567]) | **FAIL** (AUROC=0.486, CI [0.345,0.576]) | **FAIL** (AUROC=0.463, CI [0.186,0.575]) |
| (d) beats baselines | **FAIL** (loses to or ties both) | **FAIL** | **FAIL** |
| (e) collinearity (reporting only) | low, |r|<0.2 | low, |r|<0.2 | low, |r|<0.2 |

**Per the pre-committed selection rule (CRITERIA.md): candidates must
pass (a) AND (b) to be eligible for selection. Zero of C1/C2/C3 pass (b)
at all — 0 of 5 worlds clear rho>0.6 for either noise or edge-blend
sensitivity, for any candidate.** No candidate was selected. The
held-out world (`world6_dense_mixed`) was consequently **never touched**
in E3 — there is nothing meaningful to evaluate there, and touching it
anyway (e.g. to "see if it looks better") would have been exactly the
kind of unprincipled test-set peeking rule 6 forbids.

## Why C1's validity AUROC is *below chance* (0.457, not just low) — diagnosed, not just reported

A quick diagnostic (world1_baseline, viewpoint 1, 4 pairs; within the
task's 3-attempt cap for this specific question) found the mechanism
directly:

```
object_a       /object_b        est_gap   true_gap  score
obj_bottle     /obj_mug         0.0797    0.0200    0.936
obj_bowl       /obj_apple       0.0293    0.0019    0.967
obj_cereal_box /obj_banana      1.2033    0.0011    0.045
obj_notebook   /obj_remote      0.9891    0.0002    0.110
```

The last two rows are genuinely touching pairs (true gap ~1mm and
~0.2mm) where C1's bbox-ring gap estimate is off by **three orders of
magnitude** (1.2m and 0.99m estimated, vs ~1mm true). This is the exact
same background-contamination mechanism found in two prior sessions'
diagnosis of `feat_boundary_sharpness`/`feat_depth_gradient`: a thin ring
sampled around a target's own bounding-box border is, in these scenes
(most of the frame is background at `camera.clip_far`), mostly facing
background, not specifically the labeled neighbor object. C1's "10th
percentile of local depth difference over the whole ring" picks up
whatever is nearest in ANY direction around the bbox — which is
frequently a background pixel far from the true touching partner, not a
measurement of the gap to that specific pair's neighbor. This is not a
coding bug (re-verified `_c1_internals`'s logic against its own
docstring — it does exactly what was specified) — it is an inherent
limitation of the mandated **classifier-free, single-target-bbox
interface**: without knowing *which direction* the neighbor of interest
lies, any ring-based gap estimate around a single bbox is a "nearest-
anything" measure, not a "gap-to-this-specific-pair" measure, and the
two are frequently very different in a cluttered scene with a
far-background wall.

**Per the module's own committed discipline** ("This is the ONLY formula
tested; not adjusted after seeing results" — stated in
`geometric_prg_error.py`'s docstring before this evaluation ran): this is
NOT patched or re-tuned in this session. It is reported as a finding and
a proposal for a follow-up session (see `MORNING_HANDOFF_4.md`'s
recommendation) — most plausibly, restricting the ring to the
image-space direction toward the actual nearest visible surface
(computable without a specific neighbor mask, e.g. via a directional
scan rather than a uniform ring) would fix this, but that is an
unverified proposal, not something implemented or tested here.

## (b) sensitivity — reported in full, not just pass/fail

Per-world Spearman rho values (noise sweep / blend sweep) ranged from
about -0.54 to +0.40 across all candidate/world combinations — weak,
inconsistent, and often the WRONG SIGN (C3 in particular shows
consistently NEGATIVE rho with noise level on 4 of 5 worlds, meaning the
"ambiguity" score tends to go DOWN as depth gets noisier, the opposite of
what a noise-sensitivity signal should do). This is consistent with the
same root cause as (c)'s failure: the ring is dominated by a
background-vs-target contrast that swamps whatever small noise-driven
change happens at the true touching interface.

## (d) baselines

No candidate beat either the constant-0.5 baseline or the raw
`boundary_sharpness` single-feature baseline with a CI excluding 0 — all
margins were negative or included 0. The single-feature baseline itself
(AUROC=0.486) is also near chance, for the same underlying reason
(`boundary_sharpness`, per two prior sessions' findings, is itself
background-contaminated).

## (e) collinearity — no flag raised, but moot

All correlations with distance and the mask-size occlusion proxy were
below 0.2 in magnitude — no collinearity risk flagged. This is somewhat
moot given none of the candidates are being recommended regardless.

## (f) DA-V2/real-depth robustness

**UNTESTED**, stated plainly per `explore/CRITERIA.md`'s pre-committed
requirement: direct DA-V2 inference remains blocked in this environment
(missing `dinov2_layers` submodule, confirmed again this session in
`explore/BASELINE.md`, not re-fixed for the same reason as the prior
session — would mean copying files from a sibling out-of-repo project
without review). Nothing in this report should be read as saying
anything about real or DA-V2 depth.

## What this does and doesn't mean

This is a negative result for THIS session's three specific candidate
designs, not proof that no classifier-free geometric `prg_edge_error` can
ever work. The diagnosed mechanism (ring/bbox sampling without knowing
the true neighbor direction inherits background contamination) is
specific and fixable-in-principle, but fixing and re-testing it is future
work, not something this session's own committed discipline (formulas
fixed before running, not adjusted after seeing results) permits doing
now.
