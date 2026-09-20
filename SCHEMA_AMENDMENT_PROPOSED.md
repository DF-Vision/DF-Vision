# SCHEMA.md — Proposed Amendment: `relation_model_version` field (drafted, NOT applied)

Drafted 2026-09-20, autonomous retrain session. `SCHEMA.md ` (note: this
filename has a trailing space, a pre-existing filesystem artifact, not a
typo here) is Person B's interface file and is marked DRAFT (not yet
frozen) in its own header — proposing here rather than editing it
directly, matching this project's discipline for another person's
not-yet-frozen file, same as the prior session's
`diagnosis/DATA_PROVENANCE_NOTE_PROPOSED.md`.

## Why this is needed

This session added a `relation_model_version` switch to
`classify_relation.py` (`"legacy"` default, or `"v2"`, read from a new
file `relation_classifier/relation_model_config.json` — NOT
`integration/prg_config.json`). `classify_relation_versioned()` now
returns `{"relation", "confidence", "model_version"}`, but
`SCHEMA.md `'s Section 3b (per-edge fields in `prg_edges`) has no field
for which model version produced a given `relation_prediction`. Without
this, a future trial log mixing legacy and v2 predictions (e.g. during an
A/B evaluation period) would be silently ambiguous about which model
produced which row.

## Proposed addition to SCHEMA.md Section 3b

```
relation_model_version: str
Which classify_relation.py model version produced this edge's
relation_prediction/relation_confidence: "legacy" (the original frozen
logistic_model.joblib + support_logistic_model.joblib, default) or "v2"
(see retrain/models/relation_binary_v2_metadata.json for that model's
provenance, training data SHA256, and CV results). Always populated,
never None -- classify_relation_versioned() always returns a
model_version string, defaulting to "legacy" even on a missing/malformed
config file.

IMPORTANT, as of 2026-09-20: v2 exists but FAILED its pre-committed
acceptance gates (retrain/GATES_RESULTS.md) and is NOT the production
default. Any row with relation_model_version="v2" in a real trial log
would mean someone manually flipped relation_classifier/relation_model_config.json
against this session's recommendation -- flag for review, don't assume
it's an improvement.
```

## Addendum — 2026-09-20 (explore/geometric-prg-error session)

A second, analogous switch was added this session:
`relation_classifier/prg_edge_error_config.json`'s `prg_edge_error_mode`
key (`"classifier"` default, or `"geometric_c1"`/`"geometric_c2"`/`"geometric_c3"`).
Same proposal shape applies — Section 3b's `prg_edge_error` field
description should note which mode produced a given value, for the same
reason as `relation_model_version` above.

```
prg_edge_error_mode: str
Which compute_prg_edge_error() implementation produced this edge's
prg_edge_error: "classifier" (the original confidence + depth-boundary-
heuristic implementation, default) or one of "geometric_c1"/"geometric_c2"/
"geometric_c3" (classifier-free candidates from
explore/scripts/geometric_prg_error.py). Always populated, defaulting to
"classifier" even on a missing/malformed config file.

IMPORTANT, as of 2026-09-20: all three geometric candidates FAILED their
pre-committed validity/sensitivity gates (explore/E3_FINDINGS.md) and are
NOT the production default. Any row with prg_edge_error_mode != "classifier"
in a real trial log would mean someone manually flipped
relation_classifier/prg_edge_error_config.json against this session's
recommendation -- flag for review.
```

Not applied for the same reason as above — `SCHEMA.md ` is Person B's
file, pending her review.
