# Pre-committed Criteria — Exploration Session (`explore/geometric-prg-error`)

Written and committed BEFORE any experiment in this session. Not changed
after seeing results. Where the task allows tightening, tightened
choices are marked; nothing is loosened from the task's stated defaults.

---

## E1 — Oracle ceiling interpretation rule (fixed before running)

Oracle features (true minimum 3D surface gap, true vertical offset, true
horizontal overlap, true contact-normal alignment) are computed directly
from SDF ground-truth geometry — independent of any depth camera,
ray-cast renderer, or `LiveDepthFeatureSource` code path. Trained with
the same leave-one-world-out CV (world1-5) and the same held-out world
(`world6_dense_mixed`, touched once, at the end) as the retrain
session's `retrain/GATES.md`.

**Interpretation table (verbatim from the task, adopted as-is):**
- Oracle succeeds (balanced accuracy > 0.85 AND each class recall > 0.6)
  => the depth-derived `LiveDepthFeatureSource` features are lossy
  relative to the true geometry; the classification problem itself is
  solvable in principle.
- Oracle ALSO fails on Contact specifically (Contact recall does not
  clear the 0.6 bar even with true geometry) => the Contact/Separate
  label definition itself (the contact tolerance in
  `extract_gazebo_contacts.py` / the AABB epsilon in
  `SDFGeometryRelationSource`) may make the classes physically close to
  indistinguishable at the label boundary, not a feature-quality problem.
  In this branch: audit the tolerance settings and report the
  distribution of true gaps for Contact-labeled vs Separate-labeled
  pairs.
- Any other combination (e.g., oracle succeeds only partially, or
  succeeds on Contact but not Support) is reported as its own finding,
  not forced into one of the two boxes above.

No threshold in this table is changed after seeing the E1 result.

---

## E2/E3 — Geometric `prg_edge_error` candidate criteria

Candidates C1 (gap-uncertainty), C2 (boundary-sharpness-ambiguity), C3
(fixed combination of C1/C2) are evaluated on the 5 non-held-out worlds
first; ONE is selected by the rule below; the selected one is evaluated
on `world6_dense_mixed` exactly once, at the end of E3.

### (a) Non-constant, in [0,1], deterministic
Binary pass/fail: computed twice with the same seed on the same input,
must be bit-identical; range must be within [0,1] on every tested pair;
must take at least 3 distinct values across the non-held-out sweep (a
"non-constant" bar loose enough to allow legitimate clustering, strict
enough to catch a broken constant-output implementation).

### (b) Sensitivity: Spearman rho > 0.6 with injected noise/blur level
Swept across `retrain/scripts/noise_model.py`'s noise levels
{0, 1, 2, 3, 5} (same levels R2's dataset generation used) and, holding
noise fixed at 0, an edge-blending-only sweep at strengths {0, 1, 2, 4, 8}
pixels (a parameter of the same noise model, isolated). Spearman rho
computed per scene across the level sweep, then averaged (not pooled
across scenes, to avoid a scene-composition confound inflating rho).
**Tightened**: also require the rho > 0.6 bar to hold on at least 4 of
the 5 non-held-out worlds individually, not just on average — a
candidate that is only sensitive on one world is not a robust noise
detector.

### (c) Validity: AUROC > 0.70, bootstrap CI lower bound > 0.55, grouped by world
"Truth" for this gate: the true minimum 3D gap (from E1's oracle
sampler) versus the SAME quantity re-estimated from noisy/ray-cast depth
(via the pair's `minimum_boundary_distance` feature, converted to metric
gap using the same scale convention `feat_minimum_boundary_distance`
already uses) — error = |estimated_gap - true_gap|; "wrong" = error
exceeds 1cm (task's stated example threshold, adopted). Bootstrap: 2000
resamples, grouped by world (resample the set of worlds with
replacement), 95% CI = [2.5, 97.5] percentile — matching this project's
established convention from the two prior sessions.

### (d) Beats dumb baselines
Baselines: (i) constant score (always 0.5, the neutral/uninformative
value `compute_prg_edge_error`'s own docstring already uses as its
"insufficient info" fallback); (ii) single-feature score = normalized
raw `boundary_sharpness` alone (the "edge sharpness" feature named in the
task), rescaled to [0,1] via a min-max fit on the non-held-out worlds
only (never on held-out data). Margin = candidate's AUROC (from c) minus
each baseline's AUROC, paired bootstrap CI (same 2000-resample,
world-grouped scheme) must exclude 0 for the CI to count as "beats."

### (e) Collinearity risk with detector-free confounders
Report Pearson AND Spearman correlation between the candidate score and
(i) a true-occlusion proxy (fraction of the target's world-frame surface
sample points, from E1's oracle sampler, that are shadowed/behind another
object from the camera's viewpoint — computed via the same ray-cast
renderer's z-buffer, so it stays camera-viewpoint-specific per row) and
(ii) object-to-camera distance (median ray-cast depth over the target's
mask). Flag |r| > 0.7 (either correlation type) as a collinearity risk
for Model D, per the task. **This is a reporting gate, not a pass/fail
gate** — a high correlation doesn't disqualify a candidate on its own,
since Gap and true occlusion are plausibly correlated by the physics of
the problem too, but it must be disclosed prominently, and is weighed
into the final candidate-selection decision below.

### (f) DA-V2 / real-depth robustness
Per `explore/BASELINE.md`: direct DA-V2 inference is BLOCKED in this
environment (missing `dinov2_layers` submodule, confirmed, not
independently re-fixed — see rationale there). **Stated plainly and
prominently in every report**: all C1/C2/C3 results in this session are
UNTESTED on DA-V2 output and UNTESTED on real camera depth. This is not
a soft caveat buried in a footnote — E3/E6's reports lead with it.

### Candidate selection rule (fixed now, before running C1/C2/C3)
Among candidates passing (a) and (b), select the one with the highest
AUROC from (c); ties (within the (c) bootstrap CI overlap) broken by
whichever has the LOWER |correlation| flagged in (e) — i.e., prefer the
less confounded candidate when validity is statistically tied. If none
of C1/C2/C3 pass both (a) and (b), none is selected, and E3 reports this
as a negative result rather than picking the least-bad option and
calling it a pass.

---

## Legacy/v2 non-negotiables (restated, unchanged from prior sessions)

- `integration/prg_config.json`'s `use_stub` is never changed on disk;
  every experiment uses an in-memory override, and `git diff` on that
  file is checked after each one.
- No legacy or v2 `.joblib`/CSV artifact is overwritten or deleted.
- `PREREG.md`, `SCHEMA.md `, `INTERFACE.md `, and Person-A's files
  (`gap_computer.py`, `conformal_gate.py`, TSDF/YOLO/occlusion files) are
  never edited in place — proposals go in separate `*_PROPOSED.md` files.
- Any new geometric `prg_edge_error` implementation ships behind an
  explicit config switch, default OFF (legacy stub/confidence-based
  behavior stays the default).
- `world6_dense_mixed` is used exactly once in E1 (oracle) and once in
  E3 (chosen candidate) — each a SEPARATE single touch, both logged.

Random seed for this session: `RANDOM_STATE = 20260920` (same as the
retrain session, for consistency across the growing body of work on this
same fixed dataset).
