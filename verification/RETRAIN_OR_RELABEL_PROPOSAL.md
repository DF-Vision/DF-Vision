# Retrain/Relabel Proposal (STOP-thread output, per V2 = STOP)

Written because V2's gate failed decisively (balanced accuracy 0.500,
one class predicted, in the one mode that had a confirmed, evidence-backed
code fix applied — see `V2_GATE.md`). No code fix was implemented; this is
a proposal only, requiring your decision before any of it is acted on.
Nothing here has been executed as a training run.

## What we actually know, stated precisely (see `V1_AUDIT.md` for evidence)

1. The frozen model predicts "Separate" on effectively 100% of tested
   pairs on ray-cast synthetic depth, regardless of ground truth
   (physics-verified or heuristic).
2. This is NOT explained by the one confirmed train/live code mismatch
   (ring kernel shape) — fixing it changes nothing (V1f).
3. `depth_gradient` has zero train/live code difference at all, so it
   cannot be the mismatch either.
4. The model's own committed `summary.txt` shows it performed reasonably
   (0.688 test accuracy, 0.803 mean CV F1) on ITS OWN reported training
   run — so the model is not inherently untrainable on some 8-feature
   input; something about what THIS session's tests feed it differs from
   what it saw at training time, but the source of that difference is not
   pinned down to a specific line of code.
5. Two things ARE independently confirmed to be wrong, just not proven
   sufficient on their own: (a) label noise — `heuristic_label` disagrees
   with physics-verified `gazebo_contact` on 60/101 of its own Contact
   calls (and now, this session, we know REAL `gazebo_contact` ground
   truth is actually obtainable — see below); (b) `feat_boundary_sharpness`/
   `feat_depth_gradient`'s interface region is background-dominated by
   design (not a mismatch, but plausibly a poor feature design regardless
   of which side of train/live it's evaluated on).
6. We do not know, and cannot know without the lost training data or new
   real captures, whether this same background-domination behavior was
   ALSO present when the real training rows were generated. If it was,
   the model may have been fit on a mostly-uninformative `boundary_sharpness`/
   `depth_gradient` signal from day one, which would be a data/feature
   design problem the ring-kernel/depth-source fixes above don't touch.

## Proposal A (recommended first step, low effort, no retraining yet): re-generate real gazebo_contact for use as the retrain TARGET

**What**: `relation_classifier/scripts/extract_gazebo_contacts.py` has now
been run successfully for the first time (this session, all 6 worlds,
`verification/gazebo_contacts_SIM/*.csv`, 652 pairs, 0 "unknown", one
flagged object). This means a proper physics-verified label set is
actually achievable, cheaply (a few minutes of headless `gz sim` per
world), for a full retrain — not just a spot-check.

**Which labels**: use `gazebo_contact` (physics-verified) as the PRIMARY
binary target, replacing or supplementing `heuristic_label`. Given the
60/101 disagreement rate found this session and last, `heuristic_label`
should not be trusted as ground truth for the Support/Contact split
either — a full re-derivation against `gazebo_contact` (extending the
support/contact split to also be physics-checked where possible, e.g. via
Z-axis contact-point positions from `extract_gazebo_contacts.py`'s
existing `num_contact_points` field) would fix the second confirmed,
independent problem at the same time.

**Expected effect**: removes the demonstrated 60/101 label-noise
contamination from the touching/not-touching target entirely. Does NOT
by itself fix the feature-design question in item 6 above.

**Effort**: small — the harness now exists and works. Main cost is
re-running `extract_relation_features.py`'s full extraction pipeline (not
just the classifier training step) against however the real feature
CSVs get regenerated, since `gazebo_contact` alone doesn't replace the
8-feature training table — it only replaces the label column.

**Risk**: none to existing artifacts if done in `experiments/`/a fresh
data path first, per rule 3.

## Proposal B (higher effort, addresses item 6 directly): investigate real Gazebo depth-camera background behavior before trusting `boundary_sharpness`/`depth_gradient` at all

**What**: capture ONE real Gazebo depth-camera render (not ray-cast) from
one of the 6 worlds (`gz sim` is confirmed working this session; the
depth-camera topics were confirmed present in `gz topic -l`'s output
during V1f's diagnostic run: `/depth_camera`, `/depth_camera_v2`, etc.),
and directly measure whether `feat_boundary_sharpness`/`feat_depth_gradient`
show the same background-domination magnitude on REAL rendered depth as
they do on this session's analytic ray-cast depth. This is a genuinely
new, concrete, boundable next step that wasn't available before this
session (the depth-camera topics are right there, already running,
already confirmed reachable).

**Expected effect**: either (a) confirms the background-domination effect
is a rendering-pipeline artifact specific to my/last session's ray-caster,
in which case the frozen model may be fine on real depth and the whole
"degenerate" finding may be a testing artifact — a materially different
conclusion than either session has reached so far; or (b) confirms real
Gazebo depth ALSO produces the same effect, in which case
`feat_boundary_sharpness`/`feat_depth_gradient` genuinely need a redesign
(restricting to the true mutual object-object boundary, as last session's
now-refuted "fix" did — but this time as a disclosed, deliberate NEW
design, agreed with Krupa, not represented as a training-definition
recovery) before any retrain is worth doing at all.

**Effort**: moderate — needs a script to subscribe to the real
`/depth_camera*` topics (pattern already exists in `scripts/depth_publisher.py`
minus the ROS2/rclpy parts, or via `gz.transport13` directly, same library
already used successfully for `extract_gazebo_contacts.py` this session),
save one frame, and run it through the same feature functions. A few
hours, not a retrain-scale effort.

**Risk**: none — read-only sensor capture from an already-running
headless sim, no hardware.

## Proposal C (largest effort, only after A and B): full retrain

Only after A (clean labels) and B (feature design resolved, if needed) are
done: re-run `train_relation_classifier.py` end-to-end on the regenerated
data, with the same group-aware CV discipline it already uses, and compare
against the current frozen model's own reported numbers (0.688 test
accuracy / 0.803 mean CV F1) as the bar to clear. This should happen in
`experiments/` first, evaluated, and only promoted to
`relation_classifier/models/*.joblib` on your explicit decision — never
silently, per rule 3.

**Also worth checking DA-V2 directly, now that a checkpoint is known to
exist** (`/home/manasi/bps-project/checkpoints/depth_anything_v2_metric_hypersim_vits_NewModified.pth`,
outside this repo — see `V1_AUDIT.md` §V1e correction): `rclpy` is still
missing, but a direct (non-ROS) Python call into `models/depth_anything_v2/dpt.py`
may not need it at all. This would let H1 (MDE-vs-GT) finally be tested
directly instead of staying INCONCLUSIVE — worth doing before or alongside
Proposal C, since it changes what "the training-time depth source" even
means for a retrain.

## What NOT to do

Do not retrain against the CURRENT training data / current label mix just
to "see if it helps" — every piece of evidence points at the labels and/or
the feature design being the problem, not the model family or
hyperparameters, and a retrain on the same inputs would very likely just
reproduce the same or a similarly-miscalibrated model, burning effort
without addressing anything this audit found.
