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

## Proposed Amendment — 2026-09-20 (proposed, pending your review) — explore/geometric-prg-error session

**`prg_edge_error`'s viability for C4-ext/Model D: four consecutive
sessions, four negative or inconclusive results — flagged for a scoping
decision, not resolved here**

This amendment exists to make a pattern visible that is easy to lose
across separate session write-ups: `prg_edge_error` (feeding C4-ext's
two-signal conformal gate and Model D per the 2026-07-14 amendment above)
has now been examined four times, by four different approaches, with no
approach producing a working signal:
1. The original classifier-confidence-based `compute_prg_edge_error()`
   depends on a binary touching classifier that predicts "Separate" on
   effectively all tested pairs (multiple sessions, see PROJECT_STATUS.md
   §1, §7, §8).
2. A full retrain attempt on clean, physics-verified labels, with
   train/serve parity enforced by construction, still failed 5 of 7
   pre-committed gates (§9; `retrain/GATES_RESULTS.md`).
3. An oracle-ceiling check (this session, §10/E1) showed the classifier's
   own INPUT FEATURES are likely lossy — true geometry recovers real
   signal the depth-derived features don't — which is informative but
   does not itself produce a working `prg_edge_error`.
4. Three classifier-free geometric candidates (this session, §10/E2-E3)
   were built and evaluated against pre-committed, un-softened gates —
   all three failed decisively (validity AUROC at or below chance).

**This is disclosed now, per this project's stated discipline of
reporting null results honestly rather than reframing them, specifically
because it bears on C4-ext's viability as a contribution.** No side is
being picked here — it may still be that a fifth approach (e.g., a fixed
version of the classifier-free candidates per `explore/E3_FINDINGS.md`'s
diagnosed mechanism, or real DA-V2/real-camera depth, both untested so
far) succeeds. But four consecutive negative/mixed results is a pattern
that should inform a deliberate, human decision about whether C4-ext
stays in scope for this paper, is reframed as an exploratory/negative-
result contribution in its own right (which PREREG.md's own C2 already
establishes as an acceptable outcome type for a different contribution),
or is dropped, rather than being chased through a fifth or sixth
autonomous session without that decision being made explicitly.

- [ ] Scoping decision on C4-ext given four consecutive sessions' results
      (date: ______)
