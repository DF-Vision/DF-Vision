# V2 — Decision Gate

## Gate conditions (from task spec, restated verbatim)

PROCEED to the fix only if BOTH hold:
1. V1a/b demonstrates a concrete definition mismatch between the
   training-time and live feature paths for at least one of the 2 suspect
   features (code or history evidence), AND
2. V1f shows corrected-feature predictions with balanced accuracy vs
   `gazebo_contact` > 0.70 AND bootstrap CI lower bound > 0.55 AND more
   than one class predicted.

## Evaluation

**Condition 1: PARTIALLY SATISFIED, technically met.** `boundary_sharpness`
has a confirmed, code-quoted mismatch (ring kernel: 5x5 ellipse at training
time vs 3x3 square live — `verification/V1_AUDIT.md` §V1a, row 1). This
satisfies the letter of "at least one of the 2 suspect features." Stated
plainly, though: this mismatch is small (~35-45% effect on
`boundary_sharpness`'s magnitude), and `depth_gradient` — the other of the
"2 suspect features" — has **no** mismatch at all (identical code and
inputs in both paths). Condition 1 should not be read as "the mismatch is
sufficient to explain the bug" — only condition 2 tests that directly, and
it does not need re-litigating here since it's about to fail outright.

**Condition 2: FAILED, decisively.**
```
balanced accuracy (pairwise/corrected mode) = 0.500   (need > 0.70)
bootstrap 95% CI lower bound                = 0.500   (need > 0.55)
classes predicted                           = 1       (need > 1)
```
All three sub-conditions fail, by a wide margin, not a borderline miss.
Source: `verification/v1f_results_SIM.csv`, `verification/V1_AUDIT.md` §V1f.

## Decision: **STOP**

Per the task's own instruction: "Do not soften the thresholds after seeing
results." The thresholds are failed outright (0.500 vs required >0.70,
and only one class predicted at all vs required >1) — there is no
borderline judgment call to make here. The fix thread stops.

**No `boundary_mode` switch will be implemented (V3 skipped).** Shipping a
`legacy`/`pairwise` toggle whose `pairwise` branch has been directly shown,
against real physics ground truth, to produce byte-for-byte the same
predictions as `legacy` on every one of 43 tested pairs would be dead code
— worse, it would misrepresent to a future reader that a "fix" exists and
was validated, when what was actually validated is that this particular
fix does nothing. V4 (independent-oracle re-validation of a fix) and V5
(end-to-end chain in pairwise mode) are consequently also skipped — there
is nothing to re-validate or chain-test.

Proceeding to `verification/RETRAIN_OR_RELABEL_PROPOSAL.md`, then V6-V7.
