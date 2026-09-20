# Morning Handoff 4 — exploration session, `explore/geometric-prg-error`

Autonomous session, 2026-09-20. Branch `explore/geometric-prg-error`,
created from `retrain/relation-classifier` @ `1e72398`. Not merged, not
pushed.

## 1. E1 verdict, in three sentences

Ground-truth geometric features (no camera, no depth, pure SDF geometry)
let a plain logistic regression recover 0.649 CV Contact recall, versus
0.00-0.27 for every depth-derived model tried in the prior retrain
session — this points toward the depth-derived features being lossy,
not toward Contact/Support/Separate being inherently indistinguishable
classes (their true-gap distributions are cleanly separated, median
1.2mm vs 35.5mm). The result is not clean confirmation, though: overall
balanced accuracy (0.826 CV) fell just short of the pre-committed 0.85
bar, and the oracle model's Contact recall collapsed to 0/4 on the
held-out world — most plausibly small-sample noise (Contact is the
rarest class, 37 of 263 CV rows) rather than a deeper problem, but this
can't be stated with full confidence on this little data. **Net: labels
are probably fine; features are probably the limiting factor; more data
on the minority classes is the most promising next lever, not a
different feature philosophy.**

## 2. Criteria table, actual numbers, PASS/FAIL

**E1 interpretation (not a pass/fail table, see verdict above):** oracle
CV balanced accuracy 0.826 (target >0.85, narrowly missed), all 3 classes
individually clear the 0.6 recall bar in CV (Separate 0.637, Contact
0.649, Support 0.703) but Contact recall is 0/4 on held-out.

**E3 candidate criteria (C1/C2/C3, all on the 5 non-held-out worlds):**

| Criterion | C1 | C2 | C3 |
|---|---|---|---|
| (a) non-constant/range/deterministic | PASS | PASS | PASS |
| (b) sensitivity (rho>0.6, ≥4/5 worlds) | FAIL (0/5) | FAIL (0/5) | FAIL (0/5) |
| (c) validity (AUROC>0.70, CI lower>0.55) | FAIL (0.457) | FAIL (0.486) | FAIL (0.463) |
| (d) beats baselines | FAIL | FAIL | FAIL |
| (e) collinearity | none flagged (|r|<0.2) | none flagged | none flagged |
| (f) DA-V2/real-depth | UNTESTED (blocked, missing `dinov2_layers`) | UNTESTED | UNTESTED |

**No candidate selected** (none pass (a) AND (b), the pre-committed
eligibility bar) — `world6_dense_mixed` was correctly never touched in
E3, since there was nothing to evaluate there once the selection rule
came back empty.

## 3. Recommendation

**Reframe the paper's C4-ext claim as an audited-negative-result finding
for now; do not adopt any candidate as the `prg_edge_error` definition.**
Neither the retrain (session 3) nor this session's classifier-free
approach (session 4) produced a working `prg_edge_error`. This is the
fourth consecutive session to examine this specific problem without a
positive result — flagged explicitly in
`PREREG_RETRAIN_AMENDMENT_PROPOSED.md`'s new addendum as something
needing your scoping decision (keep chasing it, reframe C4-ext as its own
negative-result contribution — which PREREG.md's C2 already treats as a
legitimate outcome type for a different contribution — or drop it),
rather than silently continuing into a fifth autonomous attempt.

**Risks of the recommendation**: reframing C4-ext as a negative result
changes the paper's contribution count/shape — a decision only you should
make, not something to be inferred from four sessions' worth of failed
attempts without your sign-off.

**Risks of NOT deciding**: another autonomous session might spend another
several hours re-deriving the same conclusion a fifth way, or — worse —
eventually produce a candidate that passes gates by chance/overfitting to
this small dataset and gets promoted without the scrutiny this session's
(and the retrain session's) discipline applied.

## 4. UNVERIFIED items

- Whether the diagnosed C1/C2/C3 failure mechanism (background-
  contaminated bbox rings) is actually fixable by a directional-scan
  redesign, as proposed in `explore/E3_FINDINGS.md` — proposed, not
  built or tested.
- Everything about real (non-simulation) depth — DA-V2 remains blocked
  (`dinov2_layers` submodule missing, confirmed again this session), and
  real camera capture remains undone (`explore/REAL_CAPTURE_PROTOCOL_V2.md`
  is ready but unrun).
- Whether more minority-class data would actually close E1's oracle gap
  (0.826 vs 0.85) — plausible given the sample-size pattern, not tested
  (would require generating more scenes, out of this session's scope).
- The single 0.55mm-gap "Separate"-labeled row flagged in
  `explore/E1_FINDINGS.md` — noted, not manually re-examined.

## 5. What Manasi must be told

- `integration/prg_config.json`'s `use_stub` stays `true`.
- `relation_classifier/relation_model_config.json` stays `"legacy"`
  (unchanged from the retrain session).
- `relation_classifier/prg_edge_error_config.json` (NEW this session)
  defaults to `"classifier"` — the original confidence+depth-heuristic
  `compute_prg_edge_error()` is still what runs; nothing changes for any
  existing caller.
- `prg_edge_error` still means exactly what it always meant, with the
  same reliability problems as before — now additionally confirmed that
  a completely different, classifier-free approach doesn't fix it either,
  for a specific, diagnosed reason (not just "we tried and it didn't
  work").
- The real ask for you: a decision on C4-ext's scope in the paper (see
  point 3), and — only if you want to keep pursuing this — 45-75 minutes
  with the camera and calipers per `explore/REAL_CAPTURE_PROTOCOL_V2.md`.

## 6. Claims-versus-evidence checklist

| Claim | Evidence |
|---|---|
| Oracle features recover 0.649 CV Contact recall (vs 0.00-0.27 depth-based) | `explore/E1_FINDINGS.md`, `explore/scripts/e1_oracle_ceiling.py` output |
| True gap distributions: Contact/Support ~1-2mm median, Separate ~35mm median | same file, gap-distribution table |
| Oracle sampler is camera/depth-independent | direct code read, `explore/scripts/oracle_geometry.py` imports only SDF-parsing helpers |
| C1/C2/C3 fail sensitivity (b) on 0/5 worlds each | `explore/e3_output_SIM.txt`, raw per-world rho values |
| C1/C2/C3 validity AUROC 0.457/0.486/0.463 (at/below chance) | same file |
| C1's failure mechanism: bbox-ring background contamination | `explore/E3_FINDINGS.md`, 4-pair diagnostic table showing 1000x gap-estimate errors |
| Structural independence of the new geometric module | `explore/scripts/check_structural_independence.py` output, PASS |
| `prg_edge_error_mode` switch: default unchanged, regression-tested | `tests/test_prg_edge_error_mode_regression.py`, 5/5 passing (12/12 total suite) |
| `integration/prg_config.json` untouched | `git diff HEAD --` empty, `explore/E4_VERIFICATION.md` |
| All legacy/v2 artifacts byte-identical throughout | SHA256 comparison, `explore/BASELINE.md` vs re-check in `explore/E4_VERIFICATION.md` |
| Fresh-clone + determinism confirmed | `explore/E4_VERIFICATION.md` |
| DA-V2 still blocked | `explore/BASELINE.md`, same `dinov2_layers` ModuleNotFoundError as the retrain session |

## 7. Exact commands for morning review

```bash
cat explore/E1_FINDINGS.md                          # oracle ceiling result
cat explore/E3_FINDINGS.md                           # candidate evaluation result
cat explore/e3_output_SIM.txt                        # raw numbers behind the E3 table
cat PREREG_RETRAIN_AMENDMENT_PROPOSED.md              # the C4-ext scoping-decision flag (bottom section)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -v
git diff retrain/relation-classifier..HEAD --stat -- '*.joblib' integration/prg_config.json perception-calibration-gap/PREREG.md
git log --oneline retrain/relation-classifier..HEAD
```

---

Draft message to Manasi: see `explore/DRAFT_MESSAGE_TO_MANASI.md` (not sent).
