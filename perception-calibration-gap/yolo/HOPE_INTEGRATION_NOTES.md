# HOPE Dataset Integration — Notes & Proposal

Status: PROPOSAL, not a decision. Needs your mentor's sign-off before treating
any of this as final, and needs a dated POST-HOC AMENDMENT in PREREG.md
before it affects real data collection (per PREREG.md's own rule: nothing in
Sections 1-5 gets edited directly once committed).

---

## 1. The actual constraint (confirmed via search, not assumed)

HOPE (NVIDIA's "Household Objects for Pose Estimation") consists of exactly
28 toy grocery objects, split into 4 product groups:

| Group | Objects (7+10+6+5 = 28 total) |
|---|---|
| Grocery Shelf Boxes (7) | Chocolate Pudding, Cookies, Granola Bars, Mac & Cheese, Popcorn, Raisins, Spaghetti |
| Grocery Cans (10) | Alphabet Soup, Cherries, Corn, Green Beans, Mushrooms, Peaches, Peas & Carrots, Pineapple, Tomato Sauce, Tuna |
| Fridge Fillers (6) | Butter, Cream Cheese, Milk, Orange Juice, Parmesan, Yogurt |
| Condiments (5) | BBQ Sauce, Ketchup, Mayo, Mustard, Salad Dressing |

**None of your current 5 objects (apple, tennis_ball, purse, glass, remote)
are in this list.** "Same objects can be kept" isn't literally true — your
mentor's ask most likely means: swap in HOPE grocery items for some of your
current 5, so the swapped objects have externally-available pretrained data
to test generalization against.

**Distribution mechanism:** HOPE is downloaded via Google Drive using the
`gdown` Python package, through a `setup.py` script in the official repo
(https://github.com/swtyree/hope-dataset). **This cannot be downloaded from
a network-restricted sandbox** (Google Drive isn't reachable) — it must run
on your own machine or lab PC. `download_hope_dataset.py` (this same folder)
is a ready-to-run wrapper for that — copy it to your machine and run it there.

---

## 2. Proposed swap (draft — confirm before treating as final)

Your PREREG.md already split your 5 objects into must-have (apple, remote,
glass) and nice-to-have (tennis_ball, purse). The lowest-disruption path is
to swap the **nice-to-have** objects only, leaving your core geometry/
detectability axes (spread across apple/remote/glass) untouched:

| Current object | Proposed HOPE replacement | Why this one |
|---|---|---|
| tennis_ball | **Tomato Sauce** (can) | Cylindrical, rigid, labeled — reasonably easy to detect, gives you a HOPE-overlapping object in the "can" family for the generalization check |
| purse | **Mac & Cheese** (box) | Rigid rectangular box — HOPE has 7 of these; gives spread within the shelf-box family |

**What this costs you, stated plainly:** your purse was the ONLY soft,
non-convex, irregular-shaped object in your set — dropping it removes that
geometry class entirely. All 28 HOPE objects are rigid (cans/boxes/bottles),
so no HOPE swap-in can replace that particular axis. If "soft/irregular"
geometry coverage matters to your mentor's ask, that's worth raising
explicitly rather than letting it quietly disappear.

**What stays unchanged:** apple (reference/easy-detect), remote (rigid flat
geometry outlier), glass (transparent/hard-detect stress case) — your
must-have objects, and the ones your object-selection rationale already
argued cannot be under-sampled without weakening C1's central argument.

---

## 3. What "generalization" could mean here — worth pinning down with your mentor

Two genuinely different experiments hide behind "generalization on
pre-trained data," and they lead to different engineering:

**(a) Train on HOPE's data too, alongside your self-captured images.**
Changes your YOLO fine-tuning corpus (mixes self-captured + HOPE-sourced
images for the swapped-in objects). This is a scope change to the "Fully
self-captured; no external datasets" rule in your master reference — not
inherently wrong, but it needs to be disclosed as a deviation, not silently
folded in.

**(b) Train fully on self-captured images as before, then separately
evaluate your trained pipeline against HOPE's images of the swapped objects
as an external, out-of-domain check.** This keeps your core training
self-captured (honors the original rule) and adds a new, clearly-labeled
exploratory analysis: "does Gap's predictive relationship hold when the
same object is observed by a totally different camera/lighting/background
than the one it was trained on?" This is arguably the more interesting and
defensible generalization claim, and it doesn't require re-litigating your
"self-captured only" pre-registration commitment — it's additive.

**My default recommendation, if you don't already have a strong preference
from your mentor:** (b). It's a strictly additive contribution to what you've
already built, doesn't touch your PREREG.md's core Sections 1-5 commitments
about training data, and is a genuinely interesting result either way (if
Gap's relationship holds cross-domain, that's a stronger paper; if it
doesn't, that's an honest and reportable limitation).

**Ask your mentor directly:** "Do you want HOPE images IN the training set,
or as a separate held-out generalization test?" — the answer changes which
of the two you build next.

---

## 4. Draft POST-HOC AMENDMENT for PREREG.md

Do not paste this into PREREG.md until your mentor confirms 2 and 3 above.
When ready, this goes at the bottom of PREREG.md under "7. Post-Hoc
Amendments" verbatim, dated:

```
### POST-HOC AMENDMENT — [DATE]
Reason: mentor requested a generalization check against externally
available pre-trained data (HOPE dataset).

Change: swapping tennis_ball -> HOPE "Tomato Sauce" (can) and
purse -> HOPE "Mac & Cheese" (box) in the study's object set. apple,
remote, and glass are unchanged. This removes the soft/non-convex
geometry class (previously covered by purse) from the study; no
remaining or added object covers that axis. Disclosed as a limitation.

[Choose one:]
(a) HOPE images ARE included in YOLO fine-tuning training data for the
    swapped objects, in addition to self-captured images. This is a
    deviation from Section 1's "fully self-captured, no external
    datasets" statement, made explicitly and disclosed here.
(b) HOPE images are NOT part of training. They are used ONLY as a
    separate, held-out, cross-domain evaluation set: the pipeline
    (trained entirely on self-captured images) is evaluated against
    HOPE's existing images of Tomato Sauce / Mac & Cheese to test
    whether Gap's predictive relationship (C1) holds under a different
    camera, lighting, and background than training. This is an
    additional exploratory analysis, not a replacement for the
    original pooled, object-controlled model.

New planned covariate (if (a)): image_source (self_captured | hope),
included as a fixed effect anywhere pooled models mix the two sources.
```

---

## 5. Practical next steps, in order

1. Confirm with your mentor: swap proposal (Section 2) + training-vs-eval
   framing (Section 3).
2. Once confirmed, run `download_hope_dataset.py` **on your own machine**
   (not this sandbox) to pull the actual images/meshes.
3. Physically acquire the swapped-in objects (Tomato Sauce can, Mac & Cheese
   box — both from Melissa & Doug's play grocery sets, ~$50-60 total per the
   HOPE paper, available on Amazon) so your real hardware trials use objects
   that match HOPE's.
4. Paste the finalized POST-HOC AMENDMENT into PREREG.md, dated the day you
   actually confirm it — not backdated to today.
5. Only then update `mock_data_generator.py`'s `OBJECTS` list and
   `train_yolo_scaffold.py`'s `STUDY_OBJECT_NAMES` to reflect the new
   object set, so the mock pipeline and scaffold stay consistent with
   whatever you actually decide.