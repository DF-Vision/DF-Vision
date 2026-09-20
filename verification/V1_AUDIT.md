# V1 — Audit of Last Session's Finding

Branch `fix/boundary-features`, branched from `overnight/prg-diagnosis` @ `0a51fbb`.
This audits (not assumes) last session's claim in `diagnosis/BINARY_MODEL_DIAGNOSIS.md`
that `feat_boundary_sharpness`/`feat_depth_gradient`'s interface region is
background-dominated and this explains the model's degenerate behavior.
**Bottom line, stated up front since it corrects prior work: the audit
does not support shipping a fix.** Details below; verdict table at the end.

---

## V1a. Feature-definition audit: training-time path vs LiveDepthFeatureSource

**Training-time path, identified**: `extract_world_features()`
(`relation_classifier/scripts/extract_relation_features.py:1121-1228`), called from
`main()` (line 1327) when `PIPELINE_VALIDATION_MODE=True` (line 148) sets
`DEPTH_SOURCE_KIND="gazebo_range_gt"` (line 160-163). `train_relation_classifier.py`
imports nothing from `extract_relation_features.py` directly — it just
reads whatever CSV `extract_world_features`'s caller wrote (via
`build_training_table.py` / `compare_heuristic_vs_gazebo_v3.py`'s join
logic) — confirming the CSV → `train_relation_classifier.py` link is
data-only, not a code-import link.

**Live path**: `LiveDepthFeatureSource.compute()` (lines 1274-1322).

Both paths call the exact same `feat_*` functions (same Python function
objects — verified by reading both call sites side by side). The real
question is whether their **inputs** (`ring_a`, `ring_b`, `depth`,
`points_world`, `median_depth`) are constructed identically. They are not,
in three confirmed ways:

| # | Feature(s) affected | Training-time input construction | Live input construction | Verdict |
|---|---|---|---|---|
| 1 | `boundary_sharpness`, `edge_continuity`, `occlusion_boundary_score` (all take `ring_a`/`ring_b`) | `rings[name]` loaded from a precomputed `{object}_boundary.npy` file (`extract_world_features`, line 1142, via `load_object_mask()` line 849-855). That file is produced by `scripts/export_masks.py:122-136` (`RING_WIDTH=5`; `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))`; `ring = mask - cv2.erode(mask, kernel)`) or, if missing, regenerated identically by `scripts/export_boundary_mask.py:51,77,97` (same 5x5 ellipse formula, confirmed by reading both files). | `_ring_from_mask()` (`extract_relation_features.py:1267-1271`): `cv2.erode(m, np.ones((3,3), np.uint8))`, i.e. a **3x3 SQUARE** kernel, `ring_px=3`. | **MISMATCH, CONFIRMED.** Quantified directly (`verification/scripts/quantify_ring_mismatch.py`): the training-style ring has 1.15-1.84x more pixels than the live-style ring across all 18 tested pairs, and using it in place of the live ring reduces `boundary_sharpness` by roughly 35-45% on the same depth data. This is real but **small relative to the observed 14-40 SD gap** — see V1f below for the decisive test of whether it's sufficient. |
| 2 | `depth_gradient` | `mask_a \| mask_b` directly (`extract_world_features` line 1219, calling `feat_depth_gradient(depth, mask_a, mask_b)`) — no ring involved at all. | Identical: `feat_depth_gradient(depth, mask_a, mask_b)` (`LiveDepthFeatureSource.compute()` line 1318). | **MATCH.** No ring dependency, no depth-source-kind dependency, no mismatch possible. This feature has ZERO confirmed train/live code difference. |
| 3 | `surface_normal_consistency`, `relative_height` (both take `points_world`) | `points_world = depth_to_world_points(depth, camera, ..., DEPTH_SOURCE_KIND)` where `DEPTH_SOURCE_KIND="gazebo_range_gt"` when `PIPELINE_VALIDATION_MODE=True` (lines 1153-1156) — applies the range-to-Z correction (`range_to_z_depth()`, line 641-647) before back-projecting to world points. | `LiveDepthFeatureSource.compute()` line 1299-1302: `depth_to_world_points(depth, self.camera, ..., depth_source_kind="mde_predicted_z")` — **hardcoded**, ignores whatever the actual depth source is, never applies the range-to-Z correction. | **MISMATCH, CONFIRMED** (code-level, unconditional — this branch is taken regardless of input). Whether it produces numerically wrong points depends on whether the depth array fed to `LiveDepthFeatureSource` is genuinely "already Z" (as `ray_cast_scene()`'s own docstring claims it deliberately outputs, line 108-112) or "range" (needing correction) — for THIS session's ray-cast tests specifically, the hardcoded assumption happens to be correct by the ray-caster's own design, so this mismatch, while real, does not bias this session's specific numbers. It WOULD matter for any true range-depth source (e.g. a raw Gazebo depth-camera capture) fed through the live path. |
| 4 | `minimum_boundary_distance` | `median_depth = float(np.nanmedian(depth))` (`extract_world_features` line 1185) — **whole-image** median, including background. | `median_depth = float(np.nanmedian(depth[mask_a \| mask_b]))` (`LiveDepthFeatureSource.compute()` line 1306) — **restricted to the two objects' own pixels.** | **MISMATCH, CONFIRMED** (code-level). Feeds into `metres_per_px = median_depth / camera.fx`, the px→m scale factor. Not quantified numerically in this pass (lower priority — this feature's z-score was moderate, 1.2-2.4, in last session's data, not in the 14-40 range) but a real, independently-findable difference. |
| 5 | `overlap_ratio` | `feat_overlap_ratio(mask_a, mask_b)` — bbox IoU only. | Identical call. | **MATCH.** No depth, no ring — no mismatch possible. |

**Core deliverable answer**: YES, there is at least one concrete,
code-level definition mismatch (#1: ring kernel shape/width; #3: hardcoded
depth-source-kind; #4: median_depth scope). But **for the two features
PROJECT_STATUS/last session singled out as "suspect" (`boundary_sharpness`
and `depth_gradient`), only `boundary_sharpness` has a confirmed mismatch,
and it is quantified as a ~35-45% effect, not the 14-40x (1400-4000%)
effect that would be needed to explain the observed z-scores.** `depth_gradient`
has NO mismatch at all — it is bit-for-bit identical code and inputs in
both paths (feature #2 above). This is the single most important
correction to last session's framing: **the "interface region includes
background" property is not a train/live mismatch. It is a property of
`feat_boundary_sharpness`/`feat_depth_gradient` themselves, present
identically in the training-time and live code.** Whether that shared
property behaves differently on real training-time depth data versus this
session's ray-cast depth is a DATA/rendering question (see V1c, V1f), not
a code-mismatch question — and it cannot be answered from code alone.

---

## V1b. History audit — did the definitions change since training?

```
$ git log --all -L 904,914:relation_classifier/scripts/extract_relation_features.py
$ git log --all -L 988,997:relation_classifier/scripts/extract_relation_features.py
```
Both `feat_boundary_sharpness` (lines 904-914) and `feat_depth_gradient`
(lines 988-997) were **born in their current, exact form** in the file's
only commit (`e63cf944`, 2026-07-30) and have **never been modified since**
(`git log --all --oneline -- relation_classifier/scripts/extract_relation_features.py`
returns exactly one commit).

**Which branch of the reasoning applies**: UNCHANGED since training.
**How "14-40 SD out of range" is possible despite unchanged code**: the
functions are identical, so the gap cannot come from a code change. It
must come from a **difference in the DATA fed through the same code** —
this session's ray-cast synthetic depth (background clipped to exactly
`camera.clip_far`=2.0m with zero anti-aliasing, confirmed last session)
versus whatever real Gazebo depth-camera renders actually produced the
frozen model's real training rows (unknown — the depth arrays themselves
were never committed, same as the training CSV, see V1c). This is a
strictly weaker, more honest claim than last session's "confirmed root
cause in the code" framing.

---

## V1c. Reference distribution audit

**Where do the training-distribution numbers come from?** Directly from
the frozen `relation_classifier/models/scaler.joblib`'s `.mean_` and
`.scale_` attributes — re-verified this session:

```python
>>> import joblib
>>> scaler = joblib.load("relation_classifier/models/scaler.joblib")
>>> scaler.mean_[0], scaler.scale_[0]   # boundary_sharpness
(0.10793687, 0.091109)
```

`sklearn.preprocessing.StandardScaler.fit()` computes `.mean_`/`.scale_`
as literally the per-feature mean and standard deviation of the design
matrix it was fit on — i.e. of the real 125-row training split from
`training_data_v3_metric_only.csv`. **This is NOT a docstring quoting a
lost file.** The scaler is a real, committed artifact (SHA256 recorded in
`verification/BASELINE.md`), and its `.mean_`/`.scale_` are exactly the
training distribution's mean/SD by construction. **Verdict: VERIFIED
REFERENCE, not UNVERIFIED-REFERENCE.** The 14-40 SD claim's arithmetic is
sound; what's NOT verified is which real-world rendering process produced
the raw feature values that mean/SD summarize (that provenance is what's
lost, per `diagnosis/BINARY_MODEL_DIAGNOSIS.md` T1a — a different question
than whether the reference numbers themselves are real).

---

## V1d. Validation-circularity audit

`relation_classifier/scripts/validate_live_depth_feature_source.py`'s
`my_ring()` function (lines 288-294):
```python
def my_ring(mask: np.ndarray, ring_px: int = 3) -> np.ndarray:
    """Independent reimplementation of the same simple concept as the real
    _ring_from_mask (mask minus its erosion) -- written from scratch here,
    not imported, per the anti-circularity design."""
    m = mask.astype(np.uint8)
    eroded = cv2.erode(m, np.ones((ring_px, ring_px), np.uint8))
    return m.astype(bool) & (~eroded.astype(bool))
```

**Answer, explicit**: this closed-form "ground truth" ring was derived
from the **LIVE** definition (`_ring_from_mask`, 3x3 square, `ring_px=3`
default matching exactly), not from the training-time definition
(`export_masks.py`'s 5x5 ellipse). The docstring's own words — "the same
simple concept as the real `_ring_from_mask`" — confirm this by name. So
last session's 182/0/18 validation pass rate proves `LiveDepthFeatureSource`
correctly implements **its own** ring concept, self-consistently. **It
proves nothing about whether that concept matches what training actually
used** — precisely the circularity risk flagged before this audit began.
This is a second, independent reason (beyond V1a's table) that last
session's "validated" framing overstated what was actually shown.

---

## V1e. MDE-vs-GT hypothesis, re-audited

Grepped exhaustively (repeated and extended from last session): `PIPELINE_VALIDATION_MODE`
references repo-wide, `os.environ`/`getenv`/`argparse`/`sys.argv` inside
`extract_relation_features.py`, all `.sh` files, anything named
`*run_extract*`, any README mentioning the script's usage. **Found: zero
env-var or CLI override mechanism of any kind — the flag is a pure
hardcoded module constant with no other way to have been set.** This part
is now exhaustively confirmed, not just "not found last time."

**New evidence found this session, previously missed**: the module-level
comment directly above the flag (`extract_relation_features.py:170-177`):
```
# False -> normal paper pipeline (Depth Anything V2 Metric).
# True  -> validate the geometry pipeline using Gazebo ground-truth depth.
#
# This should ONLY be enabled for the one-time pipeline validation run.
```
This is the file's own author stating the intended NORMAL/production mode
is `False` (`mde_predicted_z`), and `True` is a special one-off. Since
the file's only commit has it at `True`, and since `training_data_v3_metric_only.csv`
was never committed (so its generation circumstances are entirely outside
this commit's diff), **there is no way to know, from git, what flag value
was live when the real training CSV was generated** — it could have been
generated by an uncommitted, locally-modified copy of this file at
`PIPELINE_VALIDATION_MODE=False`, exactly matching the "one-time validation
run" framing, with the flag then left at `True` before the eventual commit.

**Correction to a factual claim in `MORNING_HANDOFF.md`/`diagnosis/BINARY_MODEL_DIAGNOSIS.md`**:
those documents said "no DA-V2 checkpoint file exists anywhere in the
repo" and treated this as a hard blocker on testing H1 directly. Re-checked
this session with a machine-wide (not just repo-wide) search:
`/home/manasi/bps-project/checkpoints/depth_anything_v2_metric_hypersim_vits_NewModified.pth`
exists, outside this git repository. The claim was true narrowly (nothing
inside `DF-Vision/`) but the broader framing ("no checkpoint exists,"
implying the direct test is entirely infeasible) was not accurate — a
checkpoint is available on this machine. `rclpy` is still genuinely
missing (`ModuleNotFoundError: No module named 'rclpy'`, re-confirmed),
which blocks `depth_publisher.py`'s specific ROS2-topic-based invocation
path, but `models/depth_anything_v2/dpt.py` is committed in this repo and
a direct (non-ROS) Python inference call may well be possible without
`rclpy` at all. **Not pursued further in this session** — it is out of
this session's V1-V7 scope (V1e asks only whether evidence exists that
the mode was ever set to `mde_predicted_z`, not to newly run DA-V2
inference) and would be a meaningfully-sized side task on top of an
already large verification pass. Flagged here so the next session does
not repeat the "no checkpoint exists" claim, and flagged again in
`RETRAIN_OR_RELABEL_PROPOSAL.md` as a concrete, likely-feasible next step.

**Verdict: INCONCLUSIVE.** (Corrected from last session's "weakened,"
which improperly implied a lean toward ruling H1 out — if anything, this
new evidence nudges slightly the OTHER way, since the author's own comment
names `False`/MDE as the normal pipeline. Neither direction is provable
from available evidence. Defaulting to INCONCLUSIVE per instructions,
since the training CSV is unrecoverable.)

---

## V1f. Re-run the rehearsal — WITH REAL gazebo_contact, WITH the train-matching fix only

Two corrections from last session, both material:
1. **Ground truth**: last session used `SDFGeometryRelationSource` (a
   static-pose heuristic) as a stand-in "ground truth" because no physics
   simulation had been run. This session, `relation_classifier/scripts/extract_gazebo_contacts.py`
   (471 lines, its own docstring says **never executed** — "This sandbox
   has no Gazebo install... none of the gz.transport13/gz.msgs10 calls
   below have been run against a live gz-sim process") was run for real,
   for the first time, against all 6 worlds. First attempt pointed at the
   wrong SDF directory (`PROJECT_ROOT/worlds`, confirmed via direct grep to
   be the STATIC/no-contact-sensor copy: 47 `<static>true</static>`, 0
   `<static>false</static>`, 0 `<sensor type='contact'>` blocks) — produced
   100% "unknown" results, which briefly looked like a Gazebo-build
   incompatibility but was actually just pointing at the wrong file.
   Corrected to `relation_classifier/training_worlds/` (the actual dynamic,
   contact-sensor-equipped copy, and `extract_gazebo_contacts.py`'s own
   original hardcoded default) — this produced real, physics-verified
   contact data for all 652 pairs across the 6 worlds with zero "unknown"
   results and one flagged drifted object (`obj_cable`, world3, a plausible
   outcome for a thin flexible object under physics). Data:
   `verification/gazebo_contacts_SIM/*.csv`.
2. **The fix under test**: last session's rehearsal (`experiments/rerun_with_corrected_interface_SIM.py`)
   recomputed `boundary_sharpness`/`depth_gradient` using a "true mutual
   A-B boundary only" region — **this was never what the training pipeline
   computed** (per V1a: the training-time interface logic is identical
   `ring_a | ring_b` code, background-inclusive, same as live). It was an
   invented, better-behaved definition, not a recovery of the real training
   formula. This session's rehearsal (`verification/scripts/v1f_rerun_with_real_gazebo_contact.py`)
   instead changes ONLY the one thing V1a actually confirmed as mismatched:
   the ring kernel (5x5 ellipse, matching `export_masks.py` exactly, vs the
   live 3x3 square) — leaving the interface-region logic (`ring_a | ring_b`)
   untouched, per the instruction "do not invent a new definition."

**Result** (n=43 pairs with real, non-"unknown" `gazebo_contact`; full
table in `verification/v1f_results_SIM.csv`):

| Mode | Balanced accuracy | Per-class recall | Bootstrap 95% CI (by world) | Classes predicted |
|---|---|---|---|---|
| Legacy (production, 3x3 square ring) | 0.500 | Separate: 1.00 (n=14), Touching: 0.00 (n=29) | [0.500, 0.500] | 1 (always "Separate") |
| Pairwise (5x5 ellipse, train-matching) | 0.500 | Separate: 1.00 (n=14), Touching: 0.00 (n=29) | [0.500, 0.500] | 1 (always "Separate") |

**The train-matching ring fix changes nothing.** Every one of the 43 pairs
is still predicted "Separate" in both modes; probabilities shift by only
a few thousandths in some rows (e.g. 1.000000 -> 0.999809), never close to
crossing the decision boundary. This is a **decisive negative result**:
the one confirmed, evidence-backed train/live code mismatch (ring kernel
shape) is real but far too small to explain the model's degenerate
behavior. Last session's apparently-successful "fix" (7/15 -> non-Separate,
60% accuracy) is now understood to have worked only because it used an
invented definition that happened to move features in a favorable
direction — not because it recovered anything the training pipeline
actually did. **This refutes last session's practical conclusion.**

---

## Verdict table

| Item | Verdict | Evidence |
|---|---|---|
| V1a: concrete definition mismatch exists for a "suspect" feature | **CONFIRMED** (boundary_sharpness, ring kernel shape) but **small** (~35-45% effect) | `quantify_ring_mismatch.py` output, this doc §V1a row 1 |
| V1a: depth_gradient has a definition mismatch | **RULED OUT** — identical code, identical inputs | this doc §V1a row 2 |
| V1a: surface_normal_consistency / relative_height have a definition mismatch | **CONFIRMED** (depth_source_kind hardcoding) but doesn't bias THIS session's ray-cast numbers, by the ray-caster's own design | this doc §V1a row 3 |
| V1a: minimum_boundary_distance has a definition mismatch | **CONFIRMED** (median_depth scope) | this doc §V1a row 4 |
| V1b: code changed since training | **RULED OUT** — single commit, never modified | git log -L output |
| V1c: reference distribution (scaler stats) is trustworthy | **CONFIRMED / VERIFIED** — real committed artifact, not a lost-file citation | scaler.joblib inspection |
| V1d: 182/0/18 validation is circular w.r.t. the ring definition | **CONFIRMED** — validator's ring is explicitly modeled on the live function, not training's | `my_ring()` docstring, quoted verbatim |
| V1e: MDE-vs-GT hypothesis | **INCONCLUSIVE** (corrected from last session's "weakened") | exhaustive grep + newly-found author comment cutting both ways |
| V1f: the confirmed mismatch, fixed, restores sensible predictions on real gazebo_contact | **RULED OUT** — 0.500 balanced accuracy, 1 class predicted, in both legacy and "fixed" modes | `v1f_results_SIM.csv` |

**Net effect on last session's central claim**: last session's "confirmed
root cause: background-dominated interface region" is **downgraded**. The
background-domination PROPERTY is real and code-provable, but it is
**not a train/live mismatch** (present identically in both paths) and the
one mismatch that IS confirmed (ring kernel shape) does not explain the
degenerate behavior when tested against real ground truth. The honest
state of knowledge is: **we know the model is degenerate on ray-cast
synthetic depth, we know the model's own scaler statistics show its
training features were much smaller-magnitude than what any tested
combination of live-path fixes produces, and we do not have a confirmed,
actionable code fix that changes this.** Whether the true explanation is
data-side (real training-time depth rendering vs. ray-cast rendering
differ in ways nobody can check without the lost data) or label-side
(the independently-confirmed heuristic/gazebo_contact 60/101 disagreement)
remains open. See `verification/V2_GATE.md` for the resulting decision.
