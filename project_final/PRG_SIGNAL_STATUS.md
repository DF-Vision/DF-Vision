# PRG Signal Status — Authoritative, Consolidated (supersedes all prior partial summaries)

This document is the single source of truth for "does a working
`prg_edge_error` signal exist." It supersedes the PRG-related sections of
`PROJECT_STATUS.md` (§1, §7-10), `MORNING_HANDOFF_3.md`,
`MORNING_HANDOFF_4.md`, `explore/E1_FINDINGS.md`, `explore/E3_FINDINGS.md`,
`project_closure/LOG.md`'s C4/C5 entries, and `project_audit/C4_AUDIT.md`
— read those for full technical detail; read only this document to know
the current state.

## Full chronological history

**1. Legacy binary "touching vs. not" classifier — decisively broken**
(`overnight/prg-diagnosis`, `fix/boundary-features` sessions). The
pre-trained model (`relation_classifier/models/logistic_model.joblib`)
predicts "Separate" on effectively 100% of tested pairs regardless of
ground truth. Root-cause hunting went through two rounds: a first
diagnosis (interface-region background contamination in
`feat_boundary_sharpness`/`feat_depth_gradient`) was later **downgraded**
by a follow-up audit (`fix/boundary-features`'s `V1_AUDIT.md`) — the
confirmed train/live mismatch (a ring-kernel shape difference) was much
smaller than first claimed, and fixing it changed nothing against real
physics-verified ground truth (`gazebo_contact`, obtained for the first
time in this same session): balanced accuracy stayed exactly 0.500,
predicting "Separate" on 43/43 pairs before and after the fix. **Decisive
STOP**, per pre-committed gates.

**2. Full retrain on physics-verified labels — decisively failed**
(`retrain/relation-classifier` session). 330 newly-generated pairs with
REAL `gazebo_contact` ground truth (not the unreliable heuristic label),
strict train/serve feature parity enforced, leave-one-world-out CV with
a genuinely held-out final world. Best of 6 candidate models: 0.667 mean
CV balanced accuracy (gate: >0.75) — and **every single candidate showed
near-zero Contact-class recall (0.00-0.27)**, consistent across
architecture and algorithm. Not one bad run; a structural limitation of
the depth-derived 8-feature set. `relation_model_version` stays
`"legacy"`.

**3. Oracle-ceiling check — feature lossiness, not label indistinguishability**
(`explore/geometric-prg-error` session, E1). Ground-truth geometric
"oracle" features (true 3D surface gap, computed directly from SDF
geometry, zero camera/depth involvement) recovered 0.649 CV Contact
recall vs. 0.00-0.27 for every depth-derived candidate — shifting the
balance of evidence toward "the depth-derived features are lossy," not
"Contact and Support are physically indistinguishable." Not clean,
though: oracle Contact recall dropped to 0/4 on the held-out world, and
overall accuracy (0.826) fell just short of the pre-committed 0.85 bar.

**4. Three classifier-free geometric candidates — decisive negative**
(`explore/geometric-prg-error` session, E2/E3). C1 (gap-uncertainty),
C2 (boundary-sharpness-ambiguity), C3 (fixed combination), all behind an
off-by-default config switch. **0 of 5 non-held-out worlds cleared the
pre-committed sensitivity bar (rho>0.6) for any candidate**; validity
AUROC at or below chance (0.457-0.486). Root cause diagnosed precisely:
sampling an omnidirectional ring around a target's own bounding box has
no knowledge of which direction a real neighbor lies, so it's dominated
by whatever is closest in ANY direction — frequently far background in a
cluttered scene, not the true touching partner. Held-out world
(`world6_dense_mixed`) correctly never touched, since no candidate
qualified.

**5. C1's gap estimator's 0.0-return bug found, then fixed** (`project/closure`,
C4). The idealized-scene sanity test found `_c1_internals()` returns
exactly 0.0 even on a clean, isolated, no-background synthetic edge — a
DIFFERENT, more fundamental bug than #4's background-contamination
mechanism. Root cause traced precisely: the ring was `dilated & ~eroded`,
straddling the bbox border with ~40-50% of its pixels trivially inside
the target's own interior (same-surface, zero local-gap by construction).
Fixed: ring changed to exterior-only (`dilated & ~mask`). Verified on 4
new idealized scenes (clean edge, two-object-touching-with-background,
separated, background-only) — all now recover the true gap exactly.

**6. Re-running the #4 evaluation after the #5 fix — unchanged, and audited
to be genuinely worse, not just noisier** (`project/closure` C4, then
independently re-verified by `project/audit`'s `C4_AUDIT.md`). Re-running
`explore/CRITERIA.md`'s UNCHANGED pre-committed criteria: sensitivity
remains exactly 0/5 worlds, identical to before. Validity AUROC for C1
dropped further, from 0.457 to 0.280 (further below chance). `project/audit`
ran the statistically correct paired, world-grouped bootstrap on this
specific change (not done by `project/closure` itself) and confirmed the
degradation is real, not an artifact of comparing two independent,
overlapping confidence intervals: 95% CI on the AUROC difference
[-0.222, -0.093], excluding zero. **Conclusion, doubly confirmed: the
0.0-return bug (#5) and the background-contamination mechanism (#4) are
DISTINCT root causes. Fixing the first did not fix, and by this specific
metric made slightly worse, the second.**

**7. A third, differently-designed classifier-free candidate — also
decisive negative** (`project/closure` C5). A depth-only,
connected-components-based neighbor-identification variant, built
specifically to fix #4's diagnosed mechanism (identifies the actual
touching/nearest region via smallest interface depth-gap, rather than
sampling omnidirectionally with no notion of "which direction"). It
genuinely works on 4 idealized synthetic scenes where #4's candidates do
not (including a two-object-touching-with-background scene that would
have defeated C1/C2/C3). Evaluated against the SAME pre-committed
`explore/CRITERIA.md` gates on the SAME 5 non-held-out worlds: still
fails decisively at the sensitivity criterion (0/5 worlds, consistently
NEGATIVE correlation with injected noise/blur — a different failure
signature than #4's mixed-sign near-zero rho, but still a clean fail).
Held-out world correctly never touched.

## Current state — one paragraph

**No working `prg_edge_error` signal exists, on any of 3 distinct design
approaches (trained classifier, ring-based classifier-free, depth-only
connected-components classifier-free), across 5 sessions of dedicated
attempts, all evaluated against the same pre-committed, never-loosened
criteria. `use_stub` remains `true` in `integration/prg_config.json`, and
this consolidation does not change that.** For this to change, at
minimum, ALL of the following would need to be true: (a) real depth data
from an actual camera (DA-V2 inference or the physical Orbbec Gemini 2,
not ray-cast/simulated depth — nothing above has ever been tested on real
depth, stated explicitly in every session that produced these results);
(b) real, ruler/caliper-measured ground-truth gaps to validate any
candidate against, per the protocol already written and ready to execute
in `explore/REAL_CAPTURE_PROTOCOL_V2.md`; (c) a new candidate design (or
a fourth attempt at one of the existing three) that clears the SAME
sensitivity and validity gates every prior attempt failed — nothing in
this repo currently suggests what that design would look like, since the
diagnosed failure mode (background/direction-ambiguity contamination) has
now defeated two different mitigation strategies. Whether to keep
attempting new designs or formally retire this contribution is an open
scoping decision for Manasi/Krupa (`project_final/RESOLUTION_LOG.md` row
11) — this document lays out the complete evidence trail for that
decision, it does not make it.
