# Pre-committed Acceptance Criteria — `project/overnight-hardening`

Written before implementing T1, T2, T3's fix, T5's gate evaluator, T6,
and T7's synthetic-recovery checks. Not loosened after seeing results,
per rule 6.

## T1 — Camera-to-arm extrinsic calibration

- Minimum 6 point-pair correspondences to attempt a solve (Kabsch/SVD on
  fewer than 6 points is numerically unstable for a 6-DOF rigid
  transform with real-world noise); **refuse below 6**.
- **Degeneracy refusal**: refuse if the camera-frame points' singular
  values (from their centered covariance) have a smallest/largest ratio
  < 0.01 (near-collinear/near-planar — insufficient 3D spread to
  constrain a full rotation).
- **Units guard**: refuse if any input point's magnitude exceeds 5.0 in
  either frame (a robot workspace and a tabletop camera view are both
  well under 5 meters; a value like 350 is almost certainly millimeters
  entered as-is).
- **Held-out residual gate**: split point pairs (min 2 held out, rest
  used to fit), refuse to SAVE the transform (still allowed to print/
  report) if held-out RMS residual > 2 cm (`--max-residual-m`,
  configurable, default 0.02).
- Output: 4x4 homogeneous transform, meters, `det(R) == +1` (proper
  rotation, not a reflection) verified before saving, YAML key
  `T_base_from_camera_optical`.

## T2 — Grasp lookup-table execution path

- Nearest-bin lookup only; **max distance to nearest bin** (config,
  default 0.05m) — beyond that, `IK_FAILURE`, never extrapolate.
- Joint-limit check against config-declared `[min, max]` per joint;
  violation → `IK_FAILURE`, never clamp silently.
- `pose_source` field always populated (`"lookup_table"` on success,
  never silently blank).
- Dry-run is the ONLY mode implemented; any motor-library call is
  asserted to occur ZERO times in tests; requires
  `--i-am-supervising` to even print a "would send" line for a
  non-dry-run invocation, and this flag is NEVER passed by anything in
  this session (rule 1).

## T3 — Checkpoint 1 template-refusal fix

- Refuse to proceed (exit non-zero, clear message) if
  `validation_scenes.csv` is byte-identical to
  `validation_scenes_template.csv`, OR if any row's `notes` field
  contains the literal substring `"REPLACE with"` (the template's own
  marker text) — either condition alone is sufficient to refuse.
- Existing 5-percentage-point MAE tolerance (`MAE_TOLERANCE = 0.05` in
  `hand_calc_validation.py`) is UNCHANGED — this task only adds a
  template-detection guard in front of it, per rule 6 ("never loosen").

## T5 — YOLO per-object acceptance gate (synthetic-only, no real training claimed)

Thresholds are placeholders for the eventual REAL evaluation (to be
revisited once real images exist) but must be non-trivial now so the
gate's PASS/FAIL logic itself is exercised by the synthetic test:
- Per-object precision > 0.5, recall > 0.5, AP50 > 0.5 — deliberately
  modest (this is a code-path test on synthetic data, not a claim about
  real detector quality).
- **An aggregate/mAP-only number is never sufficient** — the gate
  function must reject a result where the aggregate passes but any
  individual object's precision/recall/AP50 falls short, and a test
  proves this explicitly (construct a per-class result where the
  aggregate passes but one class fails, assert the gate still fails).

## T6 — PRG geometric candidates (a) sanity test and (c) changed-interface variant

Reuses `explore/CRITERIA.md`'s already pre-committed criteria (a)-(f)
UNCHANGED for evaluating any new variant — restated here for
self-containedness, not loosened:
- (a) non-constant, in [0,1], deterministic.
- (b) Spearman rho > 0.6 with injected noise/blur, on ≥4 of 5 non-held-out
  worlds.
- (c) AUROC > 0.70 (bootstrap CI lower bound > 0.55, grouped by world)
  for flagging >1cm gap-estimate error vs SDF truth.
- (d) beats constant and single-feature baselines, CI excludes 0.
- (e) collinearity with true-occlusion proxy / camera distance reported,
  flagged if |r| > 0.7 (reporting only, not pass/fail).
- (f) DA-V2/real-depth: reported as UNTESTED if still blocked (re-verify
  in T6, don't assume last session's blocker still holds).

**New for this session's T6a sanity test** (a synthetic step-edge, not
a full scene): a depth image with a single flat step edge of known pixel
location, known blur (Gaussian sigma in pixels) and known Gaussian noise
(sigma in meters) — sweep both from 0 to a stress level, assert each
candidate's raw internal signal (not the final [0,1] score, which C1
showed can partially cancel out real signal — see T6a itself for why
both are checked) is monotonically non-decreasing in each of blur and
noise (Spearman rho > 0.6 on the sweep, same bar as (b) for consistency).
If this synthetic, isolated, minimal-confound test ALSO fails, that is
strong evidence the earlier full-scene (b) failure is not merely a
scene-composition artifact.

## T7 — Synthetic statistics recovery (planted vs. null)

- **Planted-effect recovery**: at the planted effect size already
  encoded in `mock_data_generator.py` (`gap_signal_multiplier=1.0`),
  Model B's (occlusion + Gap_initial) Gap coefficient must have the
  correct sign (matching the generator's known planted direction) in
  ≥ 90% of repeated synthetic datasets (seeded, ≥ 50 repeats), and its
  magnitude must fall within a 3x band of the generator's own implied
  effect size (loose, since exact recovery isn't expected — the point is
  "recovers the right sign and roughly the right size," not exact
  parameter recovery).
- **Conformal coverage on planted data**: empirical coverage within ±5
  percentage points of the target `1-alpha`, pooled.
- **Null-data false-positive rate**: across ≥ 100 simulated null
  datasets (`gap_signal_multiplier=0.0`), the fraction with a
  significant (p<0.05) Gap coefficient must be reported honestly — no
  pass/fail bar is pre-committed here (a null-FPR check is a calibration
  report, not a gate), but it must be within a factor of 2 of the
  nominal 0.05 to be called "well-calibrated" in the write-up; outside
  that range is reported as a miscalibration finding, not hidden.

All of T1/T2/T3/T5/T6/T7's numbers above are fixed now, before any of
the corresponding implementation work below runs its first real test.
