# Proposed doc edits — training_data_v3_metric_only.csv provenance (NOT applied)

Drafted 2026-09-20, overnight session. `SCHEMA.md `, `INTERFACE.md `, and
`relation_classifier/scripts/classify_relation.py` all cite statistics or
filenames tied to `training_data_v3_metric_only.csv`, which per
`diagnosis/BINARY_MODEL_DIAGNOSIS.md` H1/T1a is confirmed unrecoverable —
never committed to git at any point, in any branch (only its non-v3,
3-of-8-feature predecessor `training_data_metric_only.csv` was ever
`git add`ed, in commit `e63cf944`). This file proposes exact edits; none
have been applied, per the task's "propose the edit, don't apply it"
instruction. (Filenames above carry a trailing space that is a real
filesystem artifact, not a typo — see OVERNIGHT_LOG.md setup note.)

---

## 1. `classify_relation.py` docstring (lines 10-20)

Currently reads, in part:

> "...trained on heuristic_label-derived Contact/Support ground truth from
> training_data_metric_only.csv (176 rows, group-aware 5-fold CV:
> F1(Support)=0.659..."

This one is about the SUPPORT sub-classifier and cites the correct,
recoverable file — no change needed there.

The binary model's OWN training data, however, is nowhere named in this
file's docstring at all — worth adding a note so a future reader doesn't
assume the binary model and the Support sub-classifier share a training
set (they don't: the binary model used the lost `training_data_v3_metric_only.csv`,
189 rows, 8 features; the Support sub-classifier used the recoverable
`training_data_metric_only.csv`, 176 usable rows, 3 features via
`_SUPPORT_FEATURE_COLUMNS`). Proposed addition, right after line 20:

```
Note on the binary (Contact/Separate) model's own training data: it was
fit on `data/training_data_v3_metric_only.csv` (189 rows after group
split: 125 train / 64 test), a file that was never committed to this
repository and is now unrecoverable (confirmed via git archaeology,
2026-09-20 -- see diagnosis/BINARY_MODEL_DIAGNOSIS.md). The frozen
logistic_model.joblib + scaler.joblib remain the only surviving record of
that training run; relation_classifier/models/summary.txt has the
original classification report and feature-importance printout. Do not
assume `training_data_metric_only.csv` (this docstring's own citation
above, for the SUPPORT sub-classifier) is the same file -- it is a
different, earlier, 3-of-8-feature dataset.
```

## 2. `SCHEMA.md ` §3b, `relation_prediction` field note

Currently reads, in part:

> "...pending Step 6 tuning against training_data_v3_metric_only.csv's
> heuristic_label..."

This describes the Support-split threshold as still-untuned and pending
validation against the lost v3 file. Per PROJECT_STATUS.md §3, this is
now STALE — the threshold was superseded by the trained
`support_logistic_model.joblib` sub-classifier in a prior session, and
that sub-classifier was trained on the recoverable
`training_data_metric_only.csv`, not v3. Proposed replacement text:

```
relation_prediction: str | None
"Contact" / "Support" / "Separate", from classify_relation(). "Contact"
and "Separate" are direct model outputs (logistic_model.joblib, trained
on physics-verified gazebo_contact, binary; its own training file,
training_data_v3_metric_only.csv, was never committed and is now
unrecoverable -- see diagnosis/BINARY_MODEL_DIAGNOSIS.md). "Support" is
NOT a model output in the SUPPORT_RELATIVE_HEIGHT_THRESHOLD sense this
section previously described -- that placeholder threshold was replaced
with a trained 3-feature Logistic Regression sub-classifier
(support_logistic_model.joblib + support_scaler.joblib, trained on
training_data_metric_only.csv's heuristic_label-derived Contact/Support
rows, group-aware 5-fold CV F1=0.659/balanced accuracy=0.706 -- see
relation_classifier/models/support_summary.txt). The binary model was
still never trained on a 3-way target -- this distinction remains
load-bearing for any downstream per-class analysis.
```

## 3. `SCHEMA.md ` §4, "NEW — Support-split threshold is untuned" open item

Currently reads:

> "NEW -- Support-split threshold is untuned. SUPPORT_RELATIVE_HEIGHT_THRESHOLD
> in classify_relation.py is currently a placeholder (-0.01)..."

This entire open item is stale for the same reason as #2 above. Proposed:
mark it CLOSED with a pointer to the replacement, rather than delete it
(SCHEMA.md's own freeze-discipline rule is amendments-only once frozen;
since this file is still marked DRAFT/not-yet-frozen per its own header,
either a direct edit or a dated revision note would be consistent with
its stated status — your call which).

```
~~NEW -- Support-split threshold is untuned...~~ SUPERSEDED (see
relation_classifier/scripts/classify_relation.py and
support_summary.txt): SUPPORT_RELATIVE_HEIGHT_THRESHOLD was replaced with
a trained sub-classifier before this field could be exercised with the
placeholder value. Any relation_prediction == "Support" logged from
current code uses the trained sub-classifier, not the old threshold.
```

## 4. `INTERFACE.md ` §5, "SUPPORT_RELATIVE_HEIGHT_THRESHOLD is a placeholder"

Same staleness, same fix — this bullet under "Open items" should either be
removed or marked superseded, same replacement text as #3.

---

**Not applied because**: these are Person B's (Krupa's) owned interface
files, and SCHEMA.md's own header says it's meant to be "reviewed and
agreed with the collaborator" before changes — an unattended overnight
session editing another person's not-yet-frozen interface doc, even to
fix a real staleness, is exactly the kind of "quietly pick a side" action
this project's stated discipline warns against. Proposing here for you or
her to apply.
