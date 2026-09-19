# PREREG.md — Proposed Amendments (drafted, NOT applied)

Drafted 2026-09-20, overnight autonomous session, branch `overnight/prg-diagnosis`.

Per the project's own rule, `perception-calibration-gap/PREREG.md` is never
edited in place once committed — amendments only, dated, appended. These
are drafted here as proposed text for you to review and paste in yourself
(or approve for a follow-up commit); `PREREG.md` itself has NOT been
touched by this session (`git diff main..HEAD -- perception-calibration-gap/PREREG.md`
is empty — confirm this yourself in the morning handoff).

---

## Proposed Amendment A — Structural independence checklist item

Suggested text to append to PREREG.md, under a new dated amendment section:

> ## Amendment — 2026-09-20 (proposed, pending your review)
>
> **Structural independence, my side — CONFIRMED**
>
> The Amendment — 2026-07-15 entry above left "Structural independence,
> MY claim (her depth estimator not consuming my detector's masks/bboxes):
> NOT YET CONFIRMED" open. Per PROJECT_STATUS.md §2 (row: "Structural
> independence (PRG side ↔ detector/TSDF/Gap side)"), this was closed in
> a prior session via a full AST-parsed import audit of every file on both
> sides (`extract_relation_features.py`, `classify_relation.py`,
> `compute_prg_edge_error.py`, `build_prg.py`, plus every Person-A file:
> the YOLO scaffold, detector validation, TSDF fusion, Gap computer,
> occlusion estimator, grasp estimator) with a transitive BFS closure on
> both sides' import graphs. Result: zero internal imports in any of the
> 10 files, zero overlap between the two sides. This checklist item can
> now be checked off.
>
> - [x] Structural-independence check run and passed (date: see git log
>       for the session that produced PROJECT_STATUS.md §2's AST audit,
>       not independently re-dated by this overnight session; audit
>       method: AST import parse + transitive closure)

**Why proposed rather than applied**: this session did not re-run the AST
audit itself (PROJECT_STATUS.md tags it `[VERIFIED THIS SESSION]` from a
prior session, trusted per this session's own instructions), and per the
project's rule PREREG.md amendments should be something you sign off on
by date, not something an unattended run silently appends.

---

## Proposed Amendment B — Object-count conflict (flagging only, NOT resolving)

This is a **three-way** conflict, not the two-way one PROJECT_STATUS.md
§5 item 4 describes — worth flagging precisely:

1. **PREREG.md §5 (original, Day 2, committed 2026-07-08)**: 5 objects —
   apple, tennis_ball, purse, glass, remote.
2. **PREREG.md's own "Amendment — 2026-07-15"**: still says "5 physical
   objects" but swaps remote→stapler ("Option A confirmed... 5 physical
   objects = apple, tennis_ball, purse, glass, stapler").
3. **INTERFACE.md §2** (a separate, reconstructed-after-loss file — see
   its own header: "RECONSTRUCTED [date] after original file was lost
   pre-commit"): "Option A (agreed default): **3** physical objects
   (apple, purse, glass)."

So "Option A" is used in PREREG.md's amendment to mean a 5-object set
with one substitution, and in INTERFACE.md to mean a 3-object subset —
same label, two different meanings, and neither cites the other. This
needs your call, not an AI's, since it affects §4's sample-size/power
analysis (per-object N targets) and which objects actually get printed
markers, hardware pre-flight, etc. Suggested (not applied) text to
append to PREREG.md:

> ## Amendment — 2026-09-20 (proposed, pending your review)
>
> **Object-count conflict flagged, NOT resolved**
>
> Three documents disagree on object count/set: PREREG.md §5 (5 objects,
> apple/tennis_ball/purse/glass/remote), PREREG.md's own 2026-07-15
> amendment (5 objects, remote→stapler swap), and INTERFACE.md §2 ("Option
> A" = 3 objects: apple/purse/glass). This was surfaced during an overnight
> diagnostic pass and is left open for a human decision — which set is
> actually current, and whether §4's sample-size targets (must-have:
> Apple/Remote/Glass) need updating if the object set has genuinely
> shrunk to 3. Not resolved here.
>
> - [ ] Object count/set reconciled across PREREG.md and INTERFACE.md
>       (date: ______)

---

## Note (not a PREREG amendment): SUPPORT_RELATIVE_HEIGHT_THRESHOLD is now moot

SCHEMA.md and INTERFACE.md both still describe `SUPPORT_RELATIVE_HEIGHT_THRESHOLD`
as an untuned placeholder pending validation. Per PROJECT_STATUS.md §3,
this was superseded in a prior session: the threshold was replaced with a
trained 3-feature logistic sub-classifier (`support_logistic_model.joblib`).
This isn't a PREREG.md matter (PREREG.md never mentions the threshold), but
SCHEMA.md §3b and §4's "Open items" list should probably be updated to
reflect this — see the separate proposal in
`diagnosis/DATA_PROVENANCE_NOTE_PROPOSED.md`, drafted alongside the
training-data-provenance note below since both touch the same two files.
