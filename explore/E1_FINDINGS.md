# E1 — Oracle Ceiling Findings

Full raw output: run of `explore/scripts/e1_oracle_ceiling.py`, both the
5-world CV and the one-time held-out-world touch happened in the same
single run (script executes CV then immediately evaluates held-out — see
the script for why this is not "peeking": the held-out block is evaluated
once, unconditionally, not re-run after inspecting CV numbers).

## Numbers

**Leave-one-world-out CV (5 worlds, n=263, world6_dense_mixed excluded):**

| Model | Binary bal. acc (mean/std) | Bootstrap 95% CI | recall(Separate) | recall(Contact) | recall(Support) |
|---|---|---|---|---|---|
| oracle_logreg | 0.8259 / 0.1352 | [0.7105, 0.9346] | 0.637 (n=135) | **0.649** (n=37) | 0.703 (n=91) |
| oracle_random_forest | 0.8092 / 0.1295 | [0.6161, 0.9118] | 0.733 (n=135) | 0.135 (n=37) | 0.637 (n=91) |

**Held-out world6_dense_mixed (n=67, touched once):**

| Model | Binary bal. acc | recall(Separate) | recall(Contact) | recall(Support) |
|---|---|---|---|---|
| oracle_logreg | 0.7727 | 0.545 (n=22) | **0.000** (n=4) | 0.707 (n=41) |
| oracle_random_forest | 0.8636 | 0.727 (n=22) | 0.000 (n=4) | **1.000** (n=41) |

**For comparison, the retrain session's DEPTH-DERIVED v2 model** (per
`retrain/r5_final_test_output_SIM.txt`, cited not re-run): CV mean 0.6668,
Contact recall 0.00-0.27 (all 6 candidates); held-out balanced accuracy
0.8212, bootstrap 95% CI [0.7256, 0.9020], n=67, Contact recall 0.500
(n=4, i.e. 2 of 4 correct).

## Applying the pre-committed interpretation rule (CRITERIA.md)

**Neither oracle model fully clears "succeeds" (bal. acc > 0.85 AND every
class recall > 0.6).** `oracle_logreg` comes close on CV (0.8259, just
under 0.85) and DOES clear 0.6 recall on all three classes in CV — most
importantly, **0.649 Contact recall, roughly 2.5-6x every depth-based
candidate's 0.00-0.27**. `oracle_random_forest` clears the balanced-
accuracy-adjacent numbers less convincingly and its Contact recall
(0.135) collapses almost as badly as the depth-based candidates did.

**Neither model "fails on Contact" in the sense the second branch of the
rule describes** (Contact indistinguishable even with true geometry) —
at least not in CV. `oracle_logreg`'s CV Contact recall (0.649) is
genuinely usable, not near-zero. However, **both models' held-out-world
Contact recall is exactly 0.000** — but this is on n=4 examples, and the
true-gap distribution (below) shows Contact and Separate ARE cleanly,
physically separated by gap distance, which argues against a tolerance/
label-definition problem and toward the held-out result being small-
sample noise (4 examples is not enough to estimate any recall reliably)
rather than a geometric-indistinguishability problem.

**Verdict, stated as its own finding since neither of CRITERIA.md's two
boxes fits cleanly**: the balance of evidence favors **"depth-derived
features are lossy"** — true geometry lets a plain logistic regression
recover a meaningfully non-zero Contact signal that no depth-derived
model in the retrain session came close to, and the true-gap
distributions for Contact vs Separate do not overlap in any way that
would suggest the labels are drawn from indistinguishable underlying
geometry. The gap that remains (0.826 vs the 0.85 target, and the n=4
held-out collapse) is most plausibly explained by **small sample size on
the minority classes** (37 Contact / 91 Support / 135 Separate — Contact
is barely a ninth of the CV training set) rather than by an inherent
geometric floor. This is a genuinely different, more optimistic
conclusion than the retrain session reached on its own — worth carrying
into the recommendation in `MORNING_HANDOFF_4.md`: **more labeled data on
the minority classes, not a fundamentally different feature approach, is
the most promising next lever**, though this can't be stated with full
confidence given the held-out result's own instability.

## True-gap distribution (Contact vs Separate vs Support)

| Class | n | mean gap | median gap | min gap | max gap |
|---|---|---|---|---|---|
| Separate | 157 | 48.9mm | 35.5mm | 0.55mm | 214.4mm |
| Contact | 41 | 1.65mm | 1.19mm | 0.48mm | 4.22mm |
| Support | 132 | 2.09mm | 0.93mm | 0.00mm | 15.0mm |

Contact and Support have **nearly identical gap distributions** (both
essentially "touching," sub-2mm median) — this confirms, as physically
expected, that gap distance is NOT what distinguishes Contact from
Support (that's a geometric-arrangement question: side-by-side vs
stacked, which is why `true_vertical_offset`/`true_horizontal_overlap`
exist as separate oracle features). Gap distance DOES cleanly separate
"touching" (Contact+Support, sub-2mm) from "Separate" (35mm+ median) —
**with one flagged exception**: the single smallest Separate-labeled gap
is 0.55mm, closer to the Contact/Support range than to the rest of the
Separate class. This one row is worth a manual look (not pursued further
here, logged as a minor flag) but does not change the overall picture —
it is one row out of 157, not a systematic pattern.

## Provenance

`explore/data/oracle_features_SIM.csv` — 330 rows, one per
(world,viewpoint,object_a,object_b) matching `retrain/data/relation_dataset_v2_SIM.csv`
exactly, with 4 oracle geometry columns added. Computed via
`explore/scripts/oracle_geometry.py`, independent of any camera/depth
code path (verified by direct code read: imports only
`extract_relation_features.parse_world_sdf`/`_rotation_matrix`/`world_aabb`
for pose/shape parsing, no `LiveDepthFeatureSource`, no `ray_cast_scene`,
no `cv2`/`open3d` depth processing of any kind).
