Pre-Registration — Calibration Gap in VLA Manipulation
Status: DRAFT — commit this with a timestamped git commit the moment it's filled in. Rule: nothing in this document may be edited after real trial data collection begins. If you need to change something after that point, add a dated "POST-HOC AMENDMENT" section at the bottom instead of editing the original text — don't rewrite history.

1. Scope Lock (Day 1)
This project studies robotic manipulation failure under perceptual occlusion. The central object of study is the Calibration Gap — the discrepancy between the robot's perceived visibility of a target and its true (ground-truth) visibility. The pipeline is organized as five contributions (C1–C5), tested on 4–6 real objects, all seen during YOLO fine-tuning (no held-out object). Object-level (never-seen) generalization and task generalization are explicitly out of scope. Where an analysis pools across objects, object_id is included as a fixed-effect covariate so the generalization claim is "Gap predicts failure across multiple trained objects, controlling for object identity" — not "Gap generalizes to unseen objects."

2. Contributions and Pre-Registered Hypotheses (Day 1)
Each contribution below has: (a) the plain-English claim, (b) the exact statistical form to be tested, (c) primary vs. exploratory status, (d) the pass/fail criterion decided now, before data exists.
C1 — Calibration Gap as an Independent Failure Predictor
Status: Primary. Claim: The Gap between perceived and true visibility predicts manipulation failure beyond what true occlusion alone predicts, and this holds while controlling for object identity. Model form:
failure ~ true_occlusion + Gap + C(object_id)          [Model A: occlusion only, no Gap]
failure ~ true_occlusion + Gap_initial + C(object_id)  [Model B]
failure ~ true_occlusion + Gap_max + C(object_id)      [Model C]

Planned checks: VIF on Gap vs. true_occlusion (flag if >10 — weakens the "independent" claim, report honestly rather than hide); ΔAUC (B/C vs. A) with bootstrap CI excluding 0; Gap coefficient p<0.05 with odds-ratio CI excluding 1; Random Forest + SHAP as a non-parametric cross-check, object_id included as a feature; per-object descriptive table of Gap's coefficient sign/direction (consistency across objects is the generalization evidence — not per-object significance, which is underpowered at ~30–50 trials/object). Pre-registered pass criterion: Gap significant in the pooled, object-controlled model AND ΔAUC bootstrap CI excludes 0 AND SHAP ranks Gap_max or Gap_initial top-2. Null result stance (decided now): if Gap adds nothing beyond occlusion, this is reported as a valid negative result, not hidden or reframed after the fact.
C2 — Post-hoc Recalibration
Status: Secondary/supporting. Claim: A learned correction mapping perceived_visibility → true_visibility recovers part of the performance ceiling, and a nonlinear (MLP) correction captures structure a per-bin linear correction misses. Model form: stratified LinearRegression per occlusion bin vs. MLPRegressor([perceived_visibility, Gap_initial, clutter_count] → corrected_visibility), both fit on the same trial logs, evaluated with common random numbers (CRN) across conditions to enable paired testing. Planned test: McNemar's test (or Cohen's h) comparing MLP-corrected vs. stratified-linear-corrected failure outcomes, paired via CRN. Pre-registered pass criterion: MLP beats stratified-linear on McNemar's test, and both sit above the uncorrected baseline and below the oracle (ground-truth-fed) success rate. If MLP does not beat linear, the reported finding is "miscalibration structure is largely linear within occlusion bins" — this is still a valid, reportable result.
C3 — Gap × Occlusion × Clutter Interaction
Status: Primary (supporting C1's mechanism). Claim: Gap moderates the compounding effect of occlusion and clutter — i.e., high Gap accelerates failure specifically as clutter increases, not just additively. Model form:
failure ~ true_occlusion * clutter_count * Gap_high + C(object_id)

where Gap_high is defined as a median split: Gap_high = 1 if Gap_initial is at or above the median of Gap_initial across the full pooled trial dataset, else 0. The median is computed once, on the frozen dataset, after data collection ends and before any C3 model is fit — it is not recomputed per object and not tuned after seeing failure outcomes. (Chosen over a fixed numeric threshold because no pilot data exists yet to justify one; median split is self-calibrating to the actual observed Gap distribution and needs no additional domain assumption.) Pre-registered pass criterion: the three-way interaction term is significant (p<0.05) and its coefficient sign matches the hypothesis (high Gap steepens the occlusion×clutter failure slope). If the interaction is null, report as a negative result — do not drop the term and re-test a simpler model without disclosing the original plan.
C4 — Conformal Failure Gate
Status: Primary. Claim: A Gap-based nonconformity score, calibrated via split conformal prediction, gives a distribution-free, finite-sample bound on the false-gate rate — at efficiency comparable to or better than a learned baseline — without requiring any failure data to train on. Procedure (pre-registered):
Nonconformity score s = Gap_initial / percentile_90(calibration_gaps), calibrated on a 70% split of successful trials only.
Target miscoverage α fixed now (e.g., α = 0.10) — do not tune α after seeing test-set results.
Empirical coverage validated on the held-out 30% test split, reported pooled and per object.
Baseline: a FAIL-Detect-style model (GMM or shallow autoencoder trained on successful-trial features only), taken through the identical conformal calibration procedure.
Exchangeability check: KS test across sessions and across objects. Pre-registered pass criterion: empirical coverage within ±5pp of the theoretical 1−α (pooled and per object); Gap's flagging efficiency in the same ballpark as the FAIL-Detect baseline (a tie counts as a win, since Gap needs no training data); flagged fraction lands in a useful range (~15–40% — near 0% or 100% means the gate isn't doing useful work). Split-discipline rule (write it down now): the baseline's training set and the conformal calibration/test split must never overlap, and this must hold within and across objects.
C5 — Active Mitigation via TSDF Re-Fusion
Status: Primary (supporting C4's practical value). Claim: When a trial is flagged by the C4 gate, an active re-observation step (TSDF fusion of a second viewpoint) measurably reduces the Gap and recovers some fraction of flagged trials from failure. Metrics (pre-registered):
Recovery rate: fraction of C4-flagged trials where post-fusion Gap drops below the gate threshold AND the trial subsequently succeeds.
Gap delta: Gap_after_fusion − Gap_before_fusion, reported as a distribution, not just a mean.
Negative-result tracking: trials where fusion increases Gap or fails to prevent failure must be explicitly counted and reported, not filtered out — this is a pre-registered requirement, not an optional audit. Pre-registered pass criterion: recovery rate is non-trivial (report the actual number — no threshold is pre-committed here beyond "greater than 0 and honestly reported," since this is a newer, more exploratory contribution than C1/C4).

3. Planned Covariates (named now, per project rule)
session_baseline_failure_rate — logged per session, included wherever session-level drift could confound results.
measured_true_visibility — continuous ground-truth occlusion measure (not the nominal/label occlusion), used as the true_occlusion term in all models above.
object_id — categorical fixed effect in every pooled model (C1, C3, C4 coverage checks). This is what makes the generalization claim ("controlling for object identity") a pre-registered analysis, not a post-hoc addition.
clutter_count — used in C3's interaction term and C2's MLP recalibration.

## 4. Planned Sample Size

Total N: ~180–240 trials across all objects.
Per-object N: ~30–50 trials.

Must-have (full N required): Apple, Remote, Glass.
Nice-to-have (acceptable to under-sample first if time runs short): Tennis ball, Purse.

### Power justification (Monte Carlo simulation, pre-registered before data collection)

Power was estimated via simulation rather than closed-form formulas, since
neither C1's model (logistic regression with a categorical object_id fixed
effect) nor C3's model (a three-way interaction in logistic regression) has
a reliable analytic power formula. Method: generate synthetic trial data
under a specified true effect size using the same detector-noise
assumptions as mock_data_generator.py, refit the exact pre-registered model
formula, and measure the empirical fraction of simulated datasets (n_reps =
150-300 per cell) where the coefficient of interest is significant (p<0.05)
in the correct direction.

**C1 (Gap_initial coefficient, Model B):** At the planned N (195) and the
conservative/realistic detector-noise assumption already used elsewhere in
this project (mock_data_generator.py's default noise scales), estimated
power is approximately 0.12-0.15 -- well below the conventional 0.80
target. Achieving 0.80 power at this effect size would require
approximately 3,000+ trials, which is not feasible given hardware and time
constraints (Section 8, Cut Strategy). However, if the true Gap effect is
at least ~2x the conservative default (a real possibility, not a
guarantee, given the conservative default was itself a guess), estimated
power at N=195 is approximately 0.79-0.80, meeting the target.

**Pre-registered stance on this uncertainty:** C1 remains a primary
contribution, but its adequate power is conditional on the true effect
size exceeding the conservative estimate. If the real trial data yields a
null or marginal C1 result, this must be interpreted alongside this
pre-registered power ceiling -- a null result at N=195 does not distinguish
between "no effect" and "underpowered study" if the true effect is at or
below the conservative estimate. This is disclosed now, before data
collection, specifically so it cannot be reinterpreted post-hoc as a
surprise.

**C3 (three-way interaction, occlusion x clutter x Gap_high):** At the
planned N (~180-240), estimated power ranges from approximately 0.20 to
0.30 across a range of plausible interaction effect sizes
(log-odds 1.0-2.0), never approaching the 0.80 target. Approximately
480-800 trials would be needed depending on the true effect size --
roughly 2-4x the planned N. Given this, C3 is explicitly re-classified
as **exploratory** rather than confirmatory: results will be reported
descriptively (coefficient sign, magnitude, and p-value) rather than
treated as a pass/fail test of the pre-registered hypothesis. This
reclassification is made now, before data collection, not after seeing
a null result.

Power analysis scripts: analysis/power_analysis_c1.py,
analysis/power_analysis_c3.py. Both scripts and their outputs are
committed to the repository for full reproducibility.

5. Object Selection (Day 2)
Fill this in before committing. Do not select objects after looking at any pilot data from them.
Objects chosen:
#
Object
Geometry role
Detectability role
Notes
1
Apple
Primary/reference — roughly spherical, smooth convex surface
Easy — distinct color (red/green), smooth matte skin, good contrast against typical backgrounds
Anchor object; all other objects are compared against this baseline
2
Tennis ball
Same rough size/shape class as apple (spherical) — geometry is not the differentiator here
Easy-to-medium — bright color aids detection, but fuzzy felt texture is a genuinely different surface than the apple's smooth skin
Deliberately kept despite similar shape to apple: tests whether Gap holds when geometry is constant but surface texture changes (affects both YOLO edge cues and depth-sensor IR return)
3
Purse
Geometry outlier — irregular, non-convex, soft-bodied, has straps/handles
Medium — fabric or leather texture, detectability depends on color/pattern (note actual color/material once confirmed)
Adds a non-rigid, non-convex shape class absent from the other four
4
Glass (drinking glass)
Geometry outlier — rigid, cylindrical, hollow
Hard — transparent/reflective surfaces are a known failure mode for RGB-D depth sensors (IR-based depth cameras like RealSense often return noisy or missing depth on glass/transparent surfaces)
Flag as anticipated hardware risk, not a surprise — this must be noted in the edge-cases/limitations section. If depth data on the glass is unusably noisy, this is a legitimate, plainly-reported limitation, not a pipeline bug to silently patch around
5
Remote
Geometry outlier — elongated, flat, rectangular, rigid
Hard-to-medium — typically dark/low-contrast plastic, small button details; confirm actual color once in hand
Most different geometry class from apple/ball/glass; complements the purse as a second geometry outlier but along a different axis (rigid+flat vs. soft+irregular)

Checklist:
[x] At least one object is a genuine geometry outlier vs. the primary object (remote, purse, glass all qualify)
[x] At least one object is visually easy to detect (apple) and at least one is visually hard (glass; remote likely medium-hard pending color confirmation)
[x] Grasp/task confirmed identical across all five objects — confirm this explicitly before Phase 2; a tennis ball, a purse, and a drinking glass have very different graspable affordances, so make sure the "same task" (e.g., a single top-down or side pinch grasp) is actually physically viable on all five before locking this. RESOLVED: see Option B amendment below (grasp strategy identical, target point object-specific).
[x] Similar-shaped pair (apple/tennis ball) has been flagged and justified above as a deliberate texture stress-test, not an oversight
Selection rationale (paper-ready draft):
We selected five objects — an apple, a tennis ball, a purse, a drinking glass, and a remote control — to span the geometry and detectability axes relevant to testing whether the Calibration Gap's predictive power generalizes across trained objects. The apple serves as the primary reference object: a smooth, convex, easily-detected shape. The tennis ball shares the apple's approximate size and convex geometry but differs sharply in surface texture (fuzzy felt vs. smooth skin), isolating texture as a variable independent of shape. The purse and remote each introduce a distinct geometry class absent from the spherical objects — a soft, irregular, non-convex body and a rigid, flat, elongated body, respectively — while the drinking glass introduces a transparent/reflective surface, a known stress case for depth-based perception pipelines. All five objects share an identical grasp task; only object identity and its associated perceptual properties vary.
Must-have vs. nice-to-have objects (decided now, not under time pressure later):
Must-have (full N required): Apple (reference), Remote (geometry outlier), Glass (detectability outlier) — these three anchor the core claim (spanning easy/hard detectability and reference/outlier geometry) and cannot be under-sampled without weakening C1's central argument.
Nice-to-have (acceptable to under-sample first if time runs short): Tennis ball, Purse — these add valuable additional spread but the core generalization claim survives on 3 objects if it must; cut here first, not evenly across all five.


6. Timestamp
This document finalized and committed on: 2026-07-08 16:00:24 +0530
Git commit hash: 75f7a441fd0f144f04f715314659c43c3e338607
git add PREREG.md
git commit -m "Pre-registration: hypotheses (C1-C5), covariates, object selection, locked before real trial data"


7. Post-Hoc Amendments
Anything changed after the commit above goes here, dated, with a reason. Never edit Sections 1–5 directly once committed.
(none yet)



## Amendment — 2026-07-14

**Pre-registered intent: Dual-signal extension (Model D + two-signal conformal gate)**

Before any dual-signal data exists, we state plainly: Model D
(`failure ~ true_occlusion + Gap + PRG_edge_error + C(object_id)`) and the
two-signal conformal gate (max-combination nonconformity score over Gap and
PRG_edge_error) will be evaluated once the collaborator's external
`compute_prg_edge_error(rgb, depth, target_bbox) -> float` function is
available AND has passed its structural-independence check (confirming the
boundary-preserving depth estimator does not consume the object detector's
masks or bounding boxes).

This timestamps intent prior to any real dual-signal result, per the
master reference's Section 6 (Week 0, Day 4) guidance, to protect against
a "chosen after seeing the data" critique later.

- [ ] Structural-independence check run and passed (date: ______)
- [ ] Model D fit on frozen dataset (date: ______)
- [ ] Two-signal gate coverage/efficiency compared against single-signal gate (date: ______)

## Amendment — 2026-07-14

**Grasp protocol resolution: Option B (functionally identical grasp)**

The Section 5 checklist item "Grasp/task confirmed identical across all
five objects" is resolved as follows: the grasp STRATEGY is held
identical across all five objects (top-down approach, parallel-jaw pinch,
same frozen manipulation policy, same tolerance parameters), but the
exact grasp TARGET POINT is object-specific, fixed once per object before
any real trial data collection begins, and never adjusted mid-study or
after seeing trial outcomes.

Rationale: a literally identical 3D target point (e.g., geometric
bounding-box centroid) across geometrically dissimilar objects risks
producing grasp-affordance failures unrelated to perception (e.g., a
centroid-aimed grasp landing on empty space between a purse's straps, or
inside a glass's hollow interior) — contaminating the `failure` outcome
variable with noise unrelated to Gap/occlusion, which is what C1 and C4
are actually designed to measure. Per-object baseline differences
introduced by this design are absorbed by the `object_id` fixed-effect
covariate already present in every pooled model (C1, C3, C4 coverage
checks), consistent with this project's existing statistical design.

Exact per-object grasp target points will be finalized and documented
here (dated) once physical hardware is available for direct testing,
before Phase 2 (real hardware pre-flight, Weeks 1-2) begins. This
amendment locks the STRATEGY now; per-object COORDINATES follow once
verifiable.

- [x] Grasp strategy locked: top-down, parallel-jaw pinch, frozen policy,
      identical across all five objects
- [ ] Per-object grasp target points finalized and documented (pending
      hardware availability)
## Amendment — 2026-07-14

**HOPE dataset — working decision, pending mentor confirmation**

Option (a) adopted as the working default: the previously-downloaded HOPE
dataset is held out entirely from every trained/fitted component of this
project (detector training, TSDF fusion, Gap computation, any statistical
model or conformal gate). It may optionally be used later, once the
detector is finalized, purely as a post-hoc cross-domain detection-quality
check (precision/recall on any overlapping object categories), reported as
a minor supplementary aside if pursued at all. Because it never touches
anything the paper's results depend on, this is not treated as a disclosed
deviation from the self-captured-data design.

Mentor confirmation requested 2026-07-14; pending as of this writing.

- [x] Working decision adopted: option (a), held out entirely
- [ ] Mentor confirmation received (date: ______)

## Amendment — 2026-07-14

**Checkpoint 1 (ground-truth validation) status: NOT STARTED**

Confirmed via `diff validation_scenes.csv validation_scenes_template.csv`
(empty diff): `validation_scenes.csv` is still byte-identical to the
starter template as of this date. The `abs_error=0.0` rows in
`validation_report.csv` reflect only that `hand_calc_validation.py`'s own
reference math agrees with `occlusion_estimator.py` on three fabricated
example scenes (explicitly labeled "REPLACE with your real measured..."
in their own notes fields) -- this has NO evidential value for Checkpoint
1 and must not be cited as ground-truth validation having occurred.

Checkpoint 1 remains gated behind real hardware availability (Weeks 3-4,
per the 6-week timeline) and requires physically-measured marker poses,
not template rows.

- [ ] Real measured scene rows entered into validation_scenes.csv (blocked
      on hardware availability)

## Amendment — 2026-07-14

**Marker physical print/measurement status: DEFERRED (not started)**

Marker PNGs have been generated (`generate_markers.py` run successfully;
`generated_markers/manifest.csv` confirmed to list all 8 markers with
correct sizes: apple/tennis_ball 3.5cm, glass/remote 4.5cm, purse 5.5cm,
occluder_A/occluder_B 5.0cm, world_origin 6.0cm, all DICT_5X5_50 @
300dpi). Physical printing and ruler measurement against this manifest
has NOT yet occurred -- deferred pending printer/hardware access, not
skipped. This must be completed before any marker is mounted on the rig
or used to collect real trial data (Weeks 1-2 hardware pre-flight
onward) -- marker-size misconfiguration silently corrupts every
downstream occlusion number and is otherwise invisible until results
look wrong.

- [ ] Markers physically printed at 100%/Actual Size scale
- [ ] Each of the 8 markers measured with a ruler and confirmed to match
      manifest.csv's size_cm (date: ______)


## Amendment — 2026-07-15

**Merge-First Protocol Day 1 — closed async (no live session)**

Per mentor's original Merge-First Protocol design, Day 1 items were closed
asynchronously with Person B rather than in a single live session, due to
a schedule change. All items below confirmed in writing (async message
exchange, 2026-07-15) unless noted otherwise:

- SCHEMA.md: FROZEN as of 2026-07-15. `prg_edge_error` (float | None,
  [0,1]) confirmed by Person B as her source of truth. `dominant_deferral_signal`,
  `voice_command_text`, `grounding_confidence` remain her system's fields;
  no conflicting claim from her side, no read/write dependency on my side.
- Object vocabulary: Option A confirmed. 5 physical objects = apple,
  tennis_ball, purse, glass, stapler (stapler substituted for remote,
  same date, personal-item reason — see running notes). Person B confirmed
  her classifier never uses object_id as input; zero effect from the swap
  or from Option A generally.
- compute_prg_edge_error() interface contract confirmed both sides:
  - Polarity: 0 = boundary reliably separated, 1 = likely unreliable
    (matches Gap's direction).
  - Failure behavior: returns None, never a sentinel float; every caller
    checks explicitly.
  - Live vs. post-hoc: post-hoc/batch only; zero latency requirement on
    either side's statistical or demo work.
- Structural independence, Person B's claim (compute_prg_edge_error()
  independent of Gap/calibration/gate output): CONFIRMED, with precision
  noted -- independence means no use of detector confidence or masks;
  target_bbox is accepted as coarse localization input, not derived from
  detector internals. This precise framing to be used in Methods, not a
  looser "no relationship to the detector at all" claim.
- Structural independence, MY claim (her depth estimator not consuming my
  detector's masks/bounding boxes): NOT YET CONFIRMED. Person B has not
  yet audited her BP-MDE integration code for this specific dependency.
  Expected confirmation date: [PENDING -- awaiting her reply to date request,
  sent 2026-07-15]. This remains the explicit blocker on C1-ext/C4-ext
  evaluation per the prior dated amendment.
- Daily integration ritual: 9:00-9:30 PM daily agreed, both rules confirmed
  (integration debt fixed next morning; interface changes require mutual
  agreement before either side edits code).


## Amendment 2026-09-27 — Power analysis correction

**This amendment supersedes, in place, the Section 4 statement that
"estimated power at N=195 is approximately 0.79-0.80, meeting the
target" (the "2x the conservative default" scenario). The original text
above is left unedited, per this document's own rule (Section 7: "Never
edit Sections 1-5 directly once committed") — this amendment is the
authoritative correction going forward.**

What Section 4 currently states (quoted verbatim):

> **C1 (Gap_initial coefficient, Model B):** At the planned N (195) and
> the conservative/realistic detector-noise assumption already used
> elsewhere in this project (mock_data_generator.py's default noise
> scales), estimated power is approximately 0.12-0.15 -- well below the
> conventional 0.80 target. Achieving 0.80 power at this effect size
> would require approximately 3,000+ trials, which is not feasible given
> hardware and time constraints (Section 8, Cut Strategy). However, if
> the true Gap effect is at least ~2x the conservative default (a real
> possibility, not a guarantee, given the conservative default was
> itself a guess), estimated power at N=195 is approximately 0.79-0.80,
> meeting the target.

What was found: `mock_data_generator.py` (the file this power estimate
is explicitly based on) was modified exactly once since this document
was written -- commit `0d74cd4` (2026-09-21, this project's `T7`
null-data bugfix). That commit changed one line:
```diff
-noise_scale = OBJECT_DETECTOR_NOISE[obj] * gap_signal_multiplier
+noise_scale = OBJECT_DETECTOR_NOISE[obj]
```
This was a genuine, necessary bugfix: at `gap_signal_multiplier=0.0`
(the "null" scenario), the OLD line zeroed BOTH the effect coefficient
AND the measurement noise, producing a degenerate constant `Gap_initial`
column. The fix was correct for that case.

However, the same line change also altered every `gap_signal_multiplier
!= 1.0` scenario, including the "2x" scenario the Section 4 passage above
depends on. An audit (`project_audit/C6_AUDIT.md`, then
`project_prereg_correction/COMMIT_DIAGNOSIS.md`) found:

1. Reconstructing the pre-`0d74cd4` generator from git history and
   re-running the power simulation at N=195, 2x reproduces the stated
   0.79-0.80 exactly (measured: 0.825 in the audit's first check;
   0.800-0.810 in reduced-scale, differently-seeded re-runs -- consistent,
   not identical, as expected from Monte Carlo noise across different
   seeds/rep-counts).
2. The two changes bundled in `0d74cd4` are separable: a version that
   fixes ONLY the null-case bug (special-casing `mult=0.0`) while leaving
   every other multiplier's behavior exactly as it was before the fix
   reproduces the same 0.79-0.80-ish number
   (`project_prereg_correction/scripts/run_isolated_check.py`: N=195,
   mult=2.0 -> power=0.800).
3. But that reproduction is not evidence the pre-fix number was correct
   -- it comes from a mechanism traced precisely: the pre-fix code's
   `gap_signal_multiplier` simultaneously (a) scaled the failure-logit
   coefficient AND (b) scaled `Gap_initial`'s own noise/variance upward.
   Both effects independently inflate apparent statistical power.
   `power_analysis_c1.py`'s own docstring -- written at the repo's
   initial commit, ~7 weeks before `0d74cd4`, unmodified since --
   explicitly states `gap_signal_multiplier` and the trial-count
   `scale_factor` should be "varied independently," with the multiplier
   representing effect-size uncertainty alone. The pre-fix coupling
   contradicted this project's own older, documented design for what
   this parameter means.
4. A real-world noise-magnitude comparison (to settle which version is
   "more realistic") could not be resolved -- the repo's only
   literature-calibrated noise reference (Nguyen et al. 2012,
   `retrain/scripts/noise_model.py`) calibrates raw depth-sensor noise in
   meters, a different physical quantity than the unitless
   visibility-fraction noise `mock_data_generator.py` actually uses. No
   valid comparison exists in this repo.

Verdict: the CURRENT (post-`0d74cd4`, decoupled) generator behavior is
the methodologically correct one, consistent with this project's own
earlier documented intent for this exact parameter. The "0.79-0.80 at
2x" fallback was not a considered, validated estimate of a
genuinely-more-optimistic-but-plausible scenario -- it was an artifact of
a since-recognized coupling bug in the same code, and nobody re-checked
this specific claim after the bug was fixed for an unrelated reason.

Corrected power table (reduced-scale reproduction, n_reps=100 per cell,
CURRENT/decoupled/correct generator; full commands and raw output:
`project_prereg_correction/scripts/full_power_table.py`, reproducible):

| N | 1x (nominal) | 2x | 3x |
|---|---|---|---|
| 90 | 0.040 | 0.150 | 0.330 |
| 120 | 0.100 | 0.160 | 0.400 |
| 180 | 0.060 | 0.310 | 0.600 |
| 195 | 0.100 | 0.340 | 0.600 |

Corrected statement for the "2x" fallback claim: under the current,
correctly-fixed generator, a 2x effect gives approximately 0.31-0.34
power at N=180-195, not 0.79-0.80. There is no multiplier in {1x, 2x,
3x} at any tested N (90-195) that reaches the 0.80 target under the
current generator except 3x at N>=180 (0.60, still short) -- no tested
cell reaches 0.80 at all under the current, correct model within this
range. `project_audit/C6_AUDIT.md` separately found (larger-scale,
1000/300-repeat runs, wider multiplier sweep) that even N=1440 (16x the
planned N) only reaches ~0.59 power at the nominal (1x) effect size.

Remaining uncertainty: none on the central technical question above (the
two changes are separable; the decoupled behavior matches this project's
own older documented intent) -- this is not an ambiguous case requiring
"both numbers as a range." The genuine remaining uncertainty this
amendment cannot resolve for anyone: whether `b_gap=2.5` (or any
multiple of it) is anywhere near the REAL Gap-failure effect size, since
zero real trials exist. That uncertainty exists regardless of which
noise model is used and is unaffected by this correction.

Recommendation: keep `0d74cd4`'s noise-model change (the decoupling) --
do not revert it. It is consistent with this project's own
pre-existing, dated documentation of what `gap_signal_multiplier` should
mean, and the pre-fix behavior's power numbers were inflated by a real,
traced mechanism (conflated effect-size and measurement-noise scaling),
not by a more-realistic noise assumption. The "Pre-registered stance on
this uncertainty" paragraph in Section 4's original text should be read
with the "meeting the target" framing removed, since no plausible tested
scenario (1x-3x, N=90-195) currently meets it.

**Decision (2026-09-27, Manasi/Krupa): the dual-signal PRG extension
(C1-ext, C4-ext, Model D) is retired for this submission; see
`project_final/PRG_SIGNAL_STATUS.md` for the full evidence trail. The
primary submission path is Gap-only (C1, C4), consistent with this
document's own pre-registered cut strategy.**