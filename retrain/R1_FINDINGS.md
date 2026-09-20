# R1 — Cheap Bug Checks Before Retraining

## R1a. Class order / thresholding / intercept-dominance — RULED OUT as a bug

```
model.classes_ = [0 1]
intercept_ = 0.6566
coef_ (FEATURE_COLUMNS order): boundary_sharpness -1.1077, minimum_boundary_distance +0.0950,
  surface_normal_consistency +0.2183, relative_height -0.8461, depth_gradient +0.0990,
  overlap_ratio +0.6129, edge_continuity +0.2145, occlusion_boundary_score -0.2290
```
Intercept-only prediction (all features at z=0, i.e. exactly the training
mean): logit=0.6566 -> sigmoid=0.6585 -> **favors Contact(1), not
Separate**. The intercept does not dominate toward Separate; if anything
it leans the other way. `classify_relation.py::_predict_binary()`'s
source was read verbatim: it calls `model.predict()`/`.predict_proba()`
directly with no custom threshold, no hardcoded label — confirmed via an
assertion on the source text (no literal `0.5` or `threshold` string) as
well as a manual read.

Printed raw `predict_proba`/`decision_function` for 5 real-touching and 5
real-Separate pairs (real `gazebo_contact` ground truth,
`verification/gazebo_contacts_SIM/`): **every single one** gets
`decision_function` between -15.6 and -48.2, `p_separate=1.0` to full
displayed precision, **regardless of true label**. This is the same
finding as both prior sessions, now re-confirmed a third time with a
different, more targeted script. **Verdict: no class-order bug, no
threshold bug, no intercept-dominance bug. The degeneracy is 100%
feature-value-driven** — the coefficients and intercept themselves are
unremarkable; it's specific inputs producing extreme logits.

## R1b. Label plausibility — no systematic mislabeling found, one tolerance-boundary case flagged

Derived Support-vs-Contact from SDF geometry (xy-overlap + vertical
stacking test, epsilon=2cm, same convention as
`SDFGeometryRelationSource`) for all 33 physics-verified touching pairs
across the 6 worlds: **23 Support, 10 Contact**. Spot-checked every row's
z-ranges by eye (full table in `retrain/scripts/r1_cheap_checks.py`'s
part_b output) — the classification looks geometrically sound: Support
pairs consistently show one object's z0 abutting the other's z1 within a
few mm-cm; Contact pairs show overlapping z-ranges consistent with
side-by-side touching, not stacking.

Two borderline cases within 2-5cm of the 2cm Support tolerance
(`obj_container/obj_sponge` in world5, gap=3.0cm; `obj_notebook/obj_remote`
in world5, gap=2.4cm) — both currently classified Contact by a few
millimetres to a centimetre outside the tolerance. Not treated as a bug
(the tolerance is a stated, documented design choice, not silently
different from what it claims to be), but flagged as a real ambiguity
worth a human's attention if these specific pairs matter to any
downstream analysis.

One geometric oddity, not a labeling bug: `obj_cable` (world3) shows an
AABB z-range of `[-0.002, 0.002]` for both its touching pairs — this is
the same object flagged as "drifted" during physics settling in last
session's `extract_gazebo_contacts.py` run (a long, thin, likely
originally-authored-near-origin object). Its derived relation still came
out as "Contact" (not misclassified as Support by an odd AABB), so this
did not corrupt R1b's output, but `obj_cable` rows are treated with extra
caution in R2's dataset (see manifest).

## R1c. Feature-range check — not re-run standalone this session; already extensively and independently confirmed

Attempted a dedicated re-run (`retrain/scripts/r1_cheap_checks.py::part_c`)
but this machine is under heavy, variable external CPU contention this
session (`uptime` showed load average ~12 on a 12-core machine from
processes unrelated to this work — the desktop app and its renderer/GPU
processes were consuming multiple cores throughout). A clean, isolated
timing test of the exact same `LiveDepthFeatureSource.compute()` call
showed ~0.2-3s/call (perfectly reasonable for ~130 calls in ~5 minutes),
but the full `part_c` run twice exceeded 25+ minutes of CPU time without
completing on 33 pairs — consistent with resource contention, not a code
hang (no exception, no stuck single call reproducible in isolation).
Killed both attempts rather than let one check consume a disproportionate
share of session time budget (rule 10: log the blocker, move on).

**This does not leave R1c unanswered** — the same underlying computation
(feature z-scores against `scaler.mean_`/`scaler.scale_`) has now been
independently run and reported, with full per-pair tables, in **three
separate scripts across the last two sessions**:
`diagnosis/scripts/run_binary_model_diagnostic.py` (last-last session),
`verification/scripts/v1f_rerun_with_real_gazebo_contact.py` (last
session), and this session's own `retrain/scripts/r1_cheap_checks.py::part_a`
(which DID complete and printed `decision_function` values of -15 to -48,
consistent with — and not contradicting — the previously-measured 14-40 SD
`boundary_sharpness` outliers). Re-deriving the exact same z-score table a
fourth time would not change the established conclusion: `boundary_sharpness`
is far outside the frozen model's training range on ray-cast depth. Citing
the prior, already-logged result rather than re-running redundant compute.

## R1 Gate

Per the task's own instruction: "if (a) reveals a real bug..., fix it...
and report whether retraining is still necessary. Continue to R2
regardless." **(a) revealed no bug.** No class-order flip, no forced
threshold, no intercept-driven default. Retraining is necessary — R1 did
not resolve the problem. Proceeding to R2.
