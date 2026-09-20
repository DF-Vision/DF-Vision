# PREREG.md — Proposed Amendment: Retrain Session Correction (drafted, NOT applied)

Drafted 2026-09-20, autonomous retrain session, branch `retrain/relation-classifier`.
Per the project's rule, `PREREG.md` is never edited in place — amendments
only, dated, appended. Proposed text below; `PREREG.md` itself has NOT
been touched this session (`git diff HEAD -- perception-calibration-gap/PREREG.md`
is empty, confirmed in `retrain/R7_VERIFICATION.md`).

---

## Proposed Amendment — 2026-09-20 (proposed, pending your review)

**Relation classifier retrain attempt: honest correction of two prior
overnight sessions' hypotheses, and a new (also negative) result**

Two prior overnight sessions (branches `overnight/prg-diagnosis` and
`fix/boundary-features`) investigated why the frozen binary touching-model
(`relation_classifier/models/logistic_model.joblib`) predicts "Separate"
on nearly every tested pair. The first proposed a "background-dominated
interface region" root cause; an audit in the second session downgraded
this to "confirmed as a real code property, but not a training-vs-live
mismatch, and not sufficient by itself to explain the degeneracy" after
testing against real, physics-verified `gazebo_contact` ground truth
(obtained by successfully running `extract_gazebo_contacts.py` for the
first time). This amendment records a third finding, from an actual
retrain attempt:

Using the exact production `LiveDepthFeatureSource` (train/serve parity
enforced by construction — no separate training-time feature code was
written), 330 newly-generated, fully-provenanced labeled pairs (real
`gazebo_contact` + geometry-derived Support/Contact split, 6 worlds x 6
camera viewpoints, randomized documented depth noise per scene — see
`retrain/data/MANIFEST.md`), and a pre-committed, un-softened set of
acceptance gates (`retrain/GATES.md`, written and committed before any
training run): **the best candidate model (of 6 tried: logistic
regression / random forest / gradient boosting, each as a two-stage
binary+Support design and as a direct 3-class classifier) reached only
~0.667 leave-one-world-out balanced accuracy (gate: >0.75) and, more
importantly, could not learn to recognize the "Contact" class at all —
recall for Contact was 0.00-0.27 across every one of the 6 candidates
tried**, while Separate and Support were both learnable to a reasonable
degree (60-80% recall in the better candidates). This was not one bad
model choice; it was consistent across architecture and algorithm choice,
suggesting the 8-feature set itself does not currently carry enough
signal to distinguish side-by-side Contact from stacked Support or no
touching at all, at least on the ray-cast synthetic depth this session
could generate. See `retrain/GATES_RESULTS.md` for the full gate table
with all G1-G7 numbers.

This does not resolve the underlying question from the prior two
sessions' amendments-in-progress (whether `feat_boundary_sharpness`/
`feat_depth_gradient` need a redesign, whether real Gazebo depth-camera
renders would behave differently than ray-cast depth) — see
`retrain/GATES_RESULTS.md`'s recommendation section for what's proposed
next. Recorded here per this project's stated discipline: findings that
bear on any pre-registered contribution get written down as they happen,
not reframed after the fact.

- [ ] Human review of `retrain/GATES_RESULTS.md`'s recommendation
      (date: ______)
