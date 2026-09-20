# Morning Handoff 2 — verification session, `fix/boundary-features`

Autonomous verification session, 2026-09-20, tasked with auditing (not
extending) last session's `overnight/prg-diagnosis` work. Branch
`fix/boundary-features`, created from `overnight/prg-diagnosis` @ `0a51fbb`.
Not merged, not pushed. Full trail: `OVERNIGHT_LOG.md` (last session) +
this session's files under `verification/`.

## 1. The V2 gate result, in three sentences

The one confirmed, code-quoted mismatch between the training-time and
live feature-extraction paths — a ring-generation kernel shape difference
(5x5 ellipse vs 3x3 square) — is real but produces only a ~35-45% effect,
nowhere near the 14-40 standard-deviation gap it was hoped to explain.
Re-running the classifier against REAL, physics-verified `gazebo_contact`
ground truth (obtained this session by successfully running, for the
first time ever, a 471-line script that had never been executed) with
that one confirmed fix applied produced a decisive negative result:
balanced accuracy 0.500, predicting "Separate" on all 43 tested pairs
regardless. **No code fix was implemented; the pre-committed decision gate
failed outright, not marginally, so per instructions no thresholds were
softened and the fix thread stopped.**

## 2. Verdict table

| Task | Status | Evidence |
|---|---|---|
| V0 — baseline snapshot | Done | `verification/BASELINE.md` |
| V1a — feature-definition audit | Done | `verification/V1_AUDIT.md` §V1a — found the ring-kernel mismatch (new, not found last session) and confirmed `depth_gradient` has NO mismatch |
| V1b — history audit | Done | Single commit, functions never modified — `verification/V1_AUDIT.md` §V1b |
| V1c — reference distribution audit | Done | Confirmed the SD reference IS the real frozen scaler, not a lost-file citation — `verification/V1_AUDIT.md` §V1c |
| V1d — validation-circularity audit | Done | Confirmed circular: the validator's "ground truth" ring is explicitly modeled on the live function, not training's — `verification/V1_AUDIT.md` §V1d |
| V1e — MDE hypothesis, re-audited | Done | Corrected to INCONCLUSIVE (new evidence cuts both ways); also corrected a factual error (a DA-V2 checkpoint DOES exist, outside the repo) — `verification/V1_AUDIT.md` §V1e |
| V1f — rehearsal re-run, real ground truth | Done | Decisive negative result — `verification/v1f_results_SIM.csv` |
| V2 — decision gate | **STOP** | `verification/V2_GATE.md` |
| V3/V4/V5 — fix implementation, oracle, end-to-end | Skipped (correctly, per STOP) | `verification/V2_GATE.md` explains why shipping a no-op switch would be worse than not shipping one |
| V6 — non-regression | Done, adapted for STOP | `verification/V6_NONREGRESSION.md` |
| V7 — docs and handoff | Done | This file, `PROJECT_STATUS.md` §8, `verification/DRAFT_MESSAGE_TO_MANASI.md` |

## 3. The one-line decision I need from you

There is no "flip the default" decision this time — nothing was changed
in production code, so there is nothing to flip. The actual decision is:
**which of the three ordered next steps in `verification/RETRAIN_OR_RELABEL_PROPOSAL.md`
to greenlight** — (A) regenerate training labels using the now-working
real `gazebo_contact` harness [recommended first, low effort, low risk],
(B) capture one real Gazebo depth-camera frame to check whether the
background-domination property also appears on real (non-ray-cast) depth
[recommended before any retrain, moderate effort, decides whether
`feat_boundary_sharpness`/`feat_depth_gradient` need a redesign], (C) a
full retrain, only after A and B [larger effort, needs your sign-off
regardless per rule 3]. My recommendation is A then B before committing
to C — retraining on the current features without knowing whether they're
salvageable would risk reproducing the same problem. Risk of NOT doing
anything: `prg_edge_error` and the binary relation classifier remain
unusable for anything beyond wiring rehearsals, same as before.

## 4. Every UNVERIFIED item

- Whether real Gazebo depth-camera renders (not ray-cast) also produce
  the background-domination effect in `boundary_sharpness`/`depth_gradient`
  — this is THE open question and Proposal B exists specifically to close
  it. Not tested this session (out of scope for V1-V7 as specified, and a
  genuinely separate, boundable task).
- The MDE-vs-GT hypothesis (H1) remains INCONCLUSIVE — not tested
  directly, still blocked on `rclpy` for the ROS2 path, though a
  checkpoint now exists and a non-ROS direct-inference path may be
  possible (untried).
- `minimum_boundary_distance`'s and `surface_normal_consistency`/`relative_height`'s
  confirmed code mismatches (V1a rows 3-4) were identified but not
  numerically quantified the way the ring-kernel mismatch was — lower
  priority since their z-scores were moderate (1.2-2.4), not extreme, in
  prior data, but genuinely unverified beyond "the code paths differ."
- The real `gazebo_contact` data obtained this session
  (`verification/gazebo_contacts_SIM/`) has not been cross-checked against
  the SDF-authored poses for correctness beyond the built-in pose-drift
  flag (only `obj_cable` in world3 flagged) — reasonable but not
  independently re-derived a second way.

## 5. What Manasi should be told

- `integration/prg_config.json`'s `use_stub` stays `true` — nothing
  changed there, nothing should change there until you decide otherwise.
- `prg_edge_error` is, right now, still not distinguishable from noise for
  any real use: the classifier feeding it predicts "Separate" on
  essentially everything tested, on any depth source verified so far, with
  or without the one confirmed code fix. Its current AUROC signal (from
  last session's sweep, 0.62) is very likely an artifact of that
  degeneracy, not evidence the two-signal design works.
- Nothing from either overnight session is citable as C1/C1-ext/C4/C4-ext
  evidence. Nothing touched hardware. See item 6 below for the specific
  claims-to-evidence trail if she wants to check any specific number
  herself.

## 6. Claims-versus-evidence checklist

| Claim | Evidence |
|---|---|
| Ring kernel differs: 5x5 ellipse (train) vs 3x3 square (live) | `scripts/export_masks.py:122-132`, `relation_classifier/scripts/extract_relation_features.py:1267-1271` — quoted verbatim in `verification/V1_AUDIT.md` §V1a |
| That mismatch is a ~35-45% effect on `boundary_sharpness` | `verification/scripts/quantify_ring_mismatch.py` output, reproduced identically in a fresh clone (`verification/V6_NONREGRESSION.md` §V6c) |
| `depth_gradient` has zero train/live mismatch | Direct code read, `extract_relation_features.py:988-997` and `:1318` — identical call, no ring | 
| `extract_gazebo_contacts.py` had never been run before this session | Its own docstring, quoted in `verification/V1_AUDIT.md` §V1f |
| Real `gazebo_contact` obtained for 652 pairs, 0 unknown | `verification/gazebo_contacts_SIM/*.csv`, generation log in this session's commits |
| Fixed-ring-kernel model still predicts Separate on 43/43 real-labeled pairs | `verification/v1f_results_SIM.csv`, script `verification/scripts/v1f_rerun_with_real_gazebo_contact.py`, reproduced in a fresh clone |
| 182/0/18 validation was circular for the ring definition | `relation_classifier/scripts/validate_live_depth_feature_source.py:288-294` (`my_ring()` docstring), quoted verbatim |
| scaler.mean_/scale_ are the real training-distribution stats, not a lost-file citation | Direct `joblib.load()` inspection, `verification/V1_AUDIT.md` §V1c |
| A DA-V2 checkpoint exists outside the repo | `find` output, `/home/manasi/bps-project/checkpoints/depth_anything_v2_metric_hypersim_vits_NewModified.pth` |
| All 5 `.joblib` files, `prg_config.json`, `PREREG.md` unchanged | SHA256 diff + `git diff` against both `unified-pipeline` and `overnight/prg-diagnosis`, `verification/V6_NONREGRESSION.md` §V6b |

Stop condition reached: V0-V7 done, V2 said STOP, V6-V7 completed as
instructed. Stopping here. Everything committed on `fix/boundary-features`.
