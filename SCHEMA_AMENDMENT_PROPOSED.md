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

## Not applied because

`SCHEMA.md ` is Person B's file, marked DRAFT pending her review per its
own header. This proposal is for her (or you) to fold in.
