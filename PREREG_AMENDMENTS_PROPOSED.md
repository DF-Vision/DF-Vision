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

---

## Proposed Amendments — 2026-09-21 (`project/overnight-hardening` session)

Five items, per this session's task list, each proposed as a dated
`PREREG.md` amendment (not applied — same discipline as above).

### 1. Gap-only (C1 + C4) as the primary path

> ## Amendment — 2026-09-21 (proposed, pending your review)
>
> **Gap-only (C1, C4) is the primary evidentiary path; C1-ext/C4-ext
> (the PRG two-signal extension) is conditional, not primary**
>
> Per the 2026-07-14 amendment above, C1-ext/C4-ext were always
> conditional on the collaborator's `compute_prg_edge_error()` passing
> structural independence AND producing a usable signal. Four
> consecutive sessions (see the dated history in item 5 below) have now
> examined the PRG relation classifier and every classifier-free
> alternative tried, without producing a working `prg_edge_error`. This
> amendment states plainly, before any further real trial data exists,
> that **the Gap-only models (Models A/B/C, the single-signal conformal
> gate) are this project's primary contribution path**, and Model D /
> the two-signal gate remain explicitly conditional — reported only if
> a working PRG signal is found later, never assumed.
>
> - [ ] Confirm Gap-only as primary (date: ______)

### 2. Model D conditional, guarded in code

> ## Amendment — 2026-09-21 (proposed, pending your review)
>
> **Model D is implemented but code-guarded, off by default**
>
> `perception-calibration-gap/analysis/model_d.py` (this session) exists
> and is fully tested (`tests/test_model_d.py`, 11/11 passing) but is
> disabled by default (`analysis/model_d_config.json: {"enabled": false}`)
> and refuses to run on zero real (non-stub) PRG rows, per this
> project's stub-exclusion discipline. This amendment records that the
> code existing does not mean the claim is being made — Model D produces
> no output unless explicitly enabled with real data present.
>
> - [ ] Acknowledge Model D's guarded/conditional status (date: ______)

### 3. Scripted-controller / 3-object descope — flagged, not decided

> ## Amendment — 2026-09-21 (proposed, pending your review)
>
> **Object count: THREE separate documents disagree (5 vs 5-with-swap
> vs 3) — flagged again, still not resolved by any autonomous session**
>
> Restated from `PREREG_RETRAIN_AMENDMENT_PROPOSED.md`'s prior finding
> (not re-litigated, just carried forward since it remains unresolved):
> `PREREG.md` §5 (original): 5 objects (apple, tennis_ball, purse, glass,
> remote). `PREREG.md`'s own 2026-07-15 amendment: 5 objects, remote→stapler
> swap. `INTERFACE.md` §2: "Option A" = 3 objects (apple, purse, glass).
> This session's own synthetic tooling (`mock_data_generator.py`,
> `session_scheduler.py`, `REAL_CAPTURE_PROTOCOL_V2.md`) all default to
> the **3-object set** (apple, purse, glass), matching `INTERFACE.md` and
> the task's own "Known context" — but this is a working assumption for
> writing runnable code, NOT a resolution of the conflict. No side is
> picked in `PREREG.md` itself.
>
> - [ ] Resolve object count/set (date: ______)

### 4. Lookup-table grasp path as infrastructure, not a claimed capability

> ## Amendment — 2026-09-21 (proposed, pending your review)
>
> **Grasp execution: a config-driven joint-space lookup table exists as
> infrastructure; it is NOT a general IK solver and NOT validated on
> real hardware**
>
> `perception-calibration-gap/grasp/lookup_table_executor.py` (this
> session, `tests/test_grasp_lookup_table.py`, 10/10 passing, dry-run
> only, no motor backend wired) provides the mechanism by which a
> perceived grasp point becomes a joint command, ONCE T1's camera-to-arm
> calibration and a hand-teleop-calibrated lookup table both exist on
> real hardware. Neither exists yet. This amendment records the
> INFRASTRUCTURE as ready-to-fill-in, not as evidence any real grasp
> capability exists.
>
> - [ ] Acknowledge lookup-table path is infrastructure only so far (date: ______)

### 5. Honest history of the failed PRG classifier and classifier-free candidates

> ## Amendment — 2026-09-21 (proposed, pending your review)
>
> **Consolidated PRG signal history, four sessions, all negative or
> inconclusive — restated here as the single canonical summary**
>
> 1. Legacy binary touching classifier: predicts "Separate" on 43/43
>    physics-verified pairs (`overnight/prg-diagnosis`,
>    `fix/boundary-features` sessions).
> 2. Full retrain on physics-verified labels, train/serve parity
>    enforced: failed gates G1, G2, G3, G4, G6 of 7
>    (`retrain/relation-classifier` session, `retrain/GATES_RESULTS.md`).
> 3. Oracle-ceiling check: true-geometry features recover meaningfully
>    better Contact-class signal (0.649 vs 0.00-0.27 recall) than any
>    depth-derived feature set tried — points at feature lossiness, not
>    label indistinguishability (`explore/geometric-prg-error` session,
>    `explore/E1_FINDINGS.md`).
> 4. Three classifier-free geometric candidates (gap-uncertainty,
>    boundary-sharpness, a fixed combination): all failed pre-committed
>    validity/sensitivity gates decisively (`explore/E3_FINDINGS.md`).
>    This session's T6a follow-up sanity test found the mechanism is
>    even more fundamental than previously known (a ring-percentile bug
>    that produces a 0.0 gap estimate even on an idealized, single-edge,
>    no-background synthetic scene) — see `project_overnight/LOG.md`'s
>    T6a entry.
>
> This history is recorded here per this project's own stated
> discipline: report null results honestly, don't hide or reframe them
> after the fact. No side is picked on whether to keep pursuing a PRG
> signal or retire the contribution — see item 1 above and
> `project_overnight/MORNING_HANDOFF.md` for the decision this needs
> from you.
>
> - [ ] Acknowledge the 4-session PRG history as the canonical record (date: ______)
