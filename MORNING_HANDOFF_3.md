# Morning Handoff 3 — retrain session, `retrain/relation-classifier`

Autonomous retrain session, 2026-09-20. Branch `retrain/relation-classifier`,
created from `fix/boundary-features` @ `44a4979`. Not merged, not pushed.

## 1. Gate table (G1-G7), actual numbers, PASS/FAIL

| Gate | Requirement | Result | PASS/FAIL |
|---|---|---|---|
| G1 | Binary touching balanced accuracy >0.75, CI lower bound >0.60 | CV mean=0.6668 (all 5 folds <0.75); held-out world6 = [FILL AFTER R5] | **FAIL** |
| G2 | Each class recall >0.50, >1 class predicted | Contact recall 0.00-0.27 across ALL 6 candidates tried | **FAIL** |
| G3 | Beats both dumb baselines, CI excludes 0 | Loses to best_single_feature_threshold (0.6668 vs 0.6672) | **FAIL** |
| G4 | ECE <0.10, Brier reported | [FILL AFTER R5] | [FILL] |
| G5 | Balanced accuracy degrades <0.10 at noise=5x (or reported prominently) | [FILL AFTER R4] | [FILL] |
| G6 | prg_edge_error non-constant, in [0,1], AUROC>0.65 | [FILL AFTER R5] | [FILL] |
| G7 | Tests pass, legacy regression identical, fresh-clone works, deterministic | 7/7 pytest passing, SHA256s unchanged, fresh-clone + determinism confirmed | **PASS** |

Full detail: `retrain/GATES_RESULTS.md`, `retrain/models/r3_training_summary.txt`,
`retrain/R7_VERIFICATION.md`.

## 2. Recommendation: **keep legacy. v2 is not ready.**

G1, G2, and G3 fail decisively — not by a small margin, and not because
of one bad model choice (all 6 candidate architecture/algorithm
combinations show the same Contact-recall collapse). `relation_model_version`
stays at its committed default, `"legacy"`, in
`relation_classifier/relation_model_config.json`. No flip is being
proposed.

**Risk of keeping legacy**: unchanged from before this session —
`logistic_model.joblib` still predicts "Separate" on almost every
real-world pair (per the two prior sessions' findings), so
`prg_edge_error` remains unreliable for any real use.

**Risk of flipping to v2 anyway**: would trade one broken behavior
(always-Separate) for a different but comparably unreliable one
(essentially never recognizes Contact, ~65-67% ceiling on the binary
question). Not an improvement worth the churn of shipping it.

**What actually changed for the better this session, independent of the
gate outcome**: (a) the lost-training-data-provenance problem is now
solved for good — `retrain/data/relation_dataset_v2_SIM.csv` +
`retrain/data/MANIFEST.md` give a fully reproducible, SHA256-logged,
seed-logged dataset with real physics-verified labels; (b) confirmed,
across 6 different model families, that the ceiling on this feature set
is real and not an artifact of one algorithm's limitations; (c) the
`relation_model_version` switch infrastructure is built, tested, and
ready to receive a genuinely better model whenever one exists, without
needing this same wiring work redone.

## 3. Every UNVERIFIED item

- **Whether real (non-ray-cast) depth changes this picture.** A DA-V2
  secondary feature set was attempted (`retrain/scripts/r2b_generate_da_v2_features.py`,
  bounded to 2 worlds x 2 viewpoints) — [FILL: result once run: either
  "produced N comparison rows, see retrain/data/relation_dataset_v2_DAV2_SIM.csv"
  or "blocked/incomplete, logged as such"]. Even if it ran, it used
  ray-cast-derived masks (no real segmenter exists yet), so it only tests
  the depth-source question, not full real-camera fidelity.
- **Whether more features, or different features, could close the
  Contact-recall gap.** Not investigated this session — would need
  feature-engineering work, not just more training data, given the
  ceiling was consistent across model families (suggesting the existing
  8 features may be the limiting factor, not the training procedure).
- **Real-hardware performance, entirely.** `retrain/REAL_CAPTURE_PROTOCOL.md`
  + `retrain/scripts/r8_analyze_real_capture.py` are ready but unrun — no
  camera was available this session.
- **G4/G5/G6's exact numbers** — [fill in once R4/R5 complete; if this
  document is being read before that happened, treat this whole gate
  table as provisional and check `retrain/GATES_RESULTS.md` for the
  latest].

## 4. What Manasi must be told

- `integration/prg_config.json`'s `use_stub` stays `true` — nothing
  changed there, nothing should change there.
- `relation_classifier/relation_model_config.json` (new file this
  session) defaults to `"legacy"` — the frozen, original binary model is
  still what runs in production. Nothing changes for any existing caller
  of `classify_relation()`.
- `prg_edge_error` still means exactly what it meant before this
  session — unreliable, for the same underlying reason (the classifier
  feeding it doesn't work well), now additionally confirmed to not be
  fixable by a straightforward retrain on clean data alone.
- The real, useful deliverable of this session is the **retraining
  infrastructure and the clean dataset**, not a new working model — worth
  saying explicitly so tonight's work isn't read as "attempted and
  failed to fix anything," when what actually happened is "built the
  tooling, ran it honestly, and it surfaced a harder problem (feature
  sufficiency) than either prior session could see."

## 5. Claims-versus-evidence checklist

| Claim | Evidence |
|---|---|
| 330 labeled pairs generated with real gazebo_contact + geometry-derived Support/Contact | `retrain/data/relation_dataset_v2_SIM.csv`, `retrain/data/MANIFEST.md` (SHA256, seeds, generator commit) |
| Train/serve parity enforced | `retrain/scripts/r2_generate_dataset.py` and `classify_relation.py`'s v2 path both call the same `LiveDepthFeatureSource.compute()` — direct code read |
| No class-order/threshold bug (R1a) | `retrain/R1_FINDINGS.md`, `retrain/scripts/r1_cheap_checks.py` output |
| Label plausibility checked, minor tolerance-boundary ambiguity flagged (R1b) | `retrain/R1_FINDINGS.md` |
| Best v2 candidate: 0.6668 CV balanced accuracy, Contact recall 0.00 | `retrain/models/r3_training_summary.txt` |
| All 6 candidates show near-zero Contact recall | same file, all 6 blocks |
| G3 fails (loses to single-feature-threshold baseline) | same file, direct number comparison |
| Legacy `classify_relation()` output bit-for-bit unchanged | `tests/test_relation_model_v2_regression.py`, 7/7 passing |
| `integration/prg_config.json` untouched | `git diff HEAD -- integration/prg_config.json` empty, `retrain/R7_VERIFICATION.md` |
| Legacy `.joblib` files byte-identical throughout | SHA256 comparison, `retrain/BASELINE.md` vs `retrain/R7_VERIFICATION.md` |
| Fresh-clone + determinism confirmed | `retrain/R7_VERIFICATION.md` |
| [G4/G5/G6 numbers] | [FILL: `retrain/GATES_RESULTS.md` once R4/R5 complete] |

## 6. Exact commands for morning review

```bash
cat retrain/GATES_RESULTS.md                        # full gate table
cat retrain/models/r3_training_summary.txt           # all 6 candidates' CV numbers
cat retrain/data/MANIFEST.md                         # dataset provenance
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_relation_model_v2_regression.py -v
git diff fix/boundary-features..HEAD --stat -- '*.joblib' integration/prg_config.json perception-calibration-gap/PREREG.md
git log --oneline fix/boundary-features..HEAD
```

---

*This document was drafted before R4 (robustness sweep) and R5 (final
held-out test) finished running. If you are reading this before those
sections show real numbers, they are still in progress in this same
session — check `retrain/GATES_RESULTS.md` for the latest committed
state, or the session's final commit message.*
