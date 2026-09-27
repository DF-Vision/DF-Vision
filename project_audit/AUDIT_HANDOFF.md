# Audit Handoff — `project/audit`

Branch `project/audit`, created from `project/closure` (`9c0d39e`). Not
merged, not pushed. This audits the `project/closure` session's claims;
it does not extend or fix anything in that session's work. All re-runs
were done from a scratch clone (`/tmp/audit_scratch_c4`, not committed)
or in-memory function calls — no committed file, `.joblib`, or config
default was touched. `git diff` confirms this below.

## Verdict 1 (Task A — the C4 "unchanged, slightly worse" claim)

**Holds up, and is actually better-supported than the closure session's
own reporting showed.** The fix (`bb6bf51`) is real, reproducible
bit-for-bit from a fresh clone, and introduces no new artifact on 3
out-of-sample synthetic scenes (camera distance, partial occlusion) this
audit constructed independently. The closure session's "measurably
worse" claim for criterion (c)'s AUROC was backed only by two
independent marginal confidence intervals — a naive check of those
alone is genuinely ambiguous (the post-fix point estimate falls inside
the pre-fix CI). This audit ran the statistically correct test instead
— a paired, world-grouped bootstrap on the AUROC difference — and found
the degradation **is** real: 95% CI on the difference [-0.222, -0.093],
excluding zero. Full detail: `project_audit/C4_AUDIT.md`.

## Verdict 2 (Task B — the C6 underpowering claim)

**The headline conclusion is real, but the closure session's framing
overstates its novelty and understates a worse, newly-tracked-down
problem.** `PREREG.md` already disclosed this exact underpowering
finding (0.12-0.15 power) before the closure session ran — it is not a
new discovery. What IS new, and more consequential: PREREG's own
optimistic escape hatch ("if the true effect is 2x the default, N=195
reaches 0.79-0.80 power") turns out to be **false under the current,
correctly-fixed data generator** — it was only ever true under a version
of `mock_data_generator.py` with an unrelated bug (fixed for good reason
by a prior session, `0d74cd4`, without anyone re-checking whether it
broke PREREG's power claim, which it did). Under the current code, a 2x
effect gives ~0.31 power, not ~0.80. There is currently no plausible
effect-size scenario, in any document reviewed, where this study is
adequately powered as planned. Full detail: `project_audit/C6_AUDIT.md`.

---

## Supporting evidence (summary — see the two audit files for full commands/output)

**Task A**: pre-fix/post-fix numbers quoted from `explore/e3_output_SIM.txt`
and `project_closure/results/c4_e3_rerun_SIM_output.txt`; independent
reproduction from a fresh clone matched bit-for-bit; a paired bootstrap
(not previously run) resolved the CI-overlap ambiguity; 4 out-of-sample
synthetic scenes (3 camera distances + 1 partial occluder) found zero
new artifacts from the fix, and one pre-existing (not fix-related)
failure mode above ~50% target occlusion.

**Task B**: `TRUE_COEFS["b_gap"] = 2.5` traced to
`mock_data_generator.py`, confirmed arbitrary/hand-specified via its own
docstring and a repo-wide grep for any citation (none found); reduced-scale
reproduction (100 vs. 1000/500 datasets) consistent with the original
headline numbers in every case; minimum detectable effect computed at
N=90 (needs `b_gap≈15`, ≈6x the default, ≈6.0x odds ratio over the 10th-
90th Gap percentile spread) and N=180 (`b_gap≈10`, ≈4x default, ≈3.3x
odds ratio); the PREREG "2x→0.80" discrepancy traced to a specific,
identified, one-line code change (`0d74cd4`) and confirmed by
reconstructing and re-running the pre-fix generator, which exactly
reproduces PREREG's stated numbers.

## Task C — Integrity re-check

**1. SHA256 of protected files vs. C0 baseline (`project_closure/LOG.md`):**
```
$ find . -iname "*.joblib" | sort | xargs sha256sum
$ sha256sum integration/prg_config.json perception-calibration-gap/PREREG.md "SCHEMA.md " "INTERFACE.md "
```
All 13 hashes **byte-identical** to the values recorded in
`project_closure/LOG.md`'s C0 entry (checked line-by-line, all match).
`integration/prg_config.json` contents: `{ "use_stub": true }`.
`git diff project/overnight-hardening..HEAD -- integration/prg_config.json`:
empty output — confirmed via git diff, not assumed.

**2. `git diff project/closure..HEAD --stat`:**
```
$ git diff project/closure..HEAD --stat
 project_audit/AUDIT_HANDOFF.md | ...
 project_audit/C4_AUDIT.md      | ...
 project_audit/C6_AUDIT.md      | ...
 3 files changed, ... insertions(+)
```
(Exact line counts will appear once this file is committed alongside the
other two — confirmed no other file appears in this diff at commit
time, see the commit log for this branch.) `git status --porcelain`
before this commit showed only `?? project_audit/` — no other file
modified, added, or deleted anywhere else in the repo.

**3. Full test suite:**
```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -v
146 passed, 1 skipped, 304 warnings in 52.06s
```
**Matches** the closure handoff's last-known count (146 passed, 1
skipped) exactly. Independently re-run, not assumed from the prior
session's own report.

---

## What I would tell Manasi and Krupa before either of them changes any decision based on the closure session's claims

1. **The PRG negative result is genuinely solid — this is the one place
   the closure session's claim held up to a stricter test than it ran
   itself.** Don't spend more time chasing this specific ring/percentile
   design; the audit found no hidden win here.

2. **Don't treat the "study is underpowered" finding as new information
   that needs a fresh decision — it's already in your own PREREG.md,
   and has been since before the closure session started.** The actual
   decision this creates is not "was C6 right" (yes) but "PREREG's own
   optimistic fallback plan doesn't work anymore" (see #3).

3. **Specifically: don't rely on PREREG's "if the real effect is 2x
   the default, N=195 will be adequately powered" sentence — it's
   currently false under the generator code you actually run today.**
   It was true once, under an old generator with a bug you (rightly)
   fixed for a different reason. Nobody re-checked the power claim after
   that fix. The real minimum detectable effect is roughly a 3.3x-6x
   odds-ratio difference between low- and high-Gap trials (at N=180 and
   N=90 respectively) — check this against any pilot trials you run
   before assuming a smaller real effect will still be detectable.

4. **The "planned N" is ambiguous across this project's own documents**
   — PREREG.md says 195, the code you'd actually run today defaults to
   90. This audit didn't resolve which is current (it's tangled up with
   the unresolved 5-vs-3-object question already flagged in
   `PREREG_AMENDMENTS_PROPOSED.md`), but whichever N you use, the
   underpowering conclusion doesn't change — only exactly how bad it is.

5. **Everything in this audit, like everything in the closure session,
   is synthetic.** Zero real trials exist. The moment real pilot data
   exists — even a small amount — re-run `power_analysis_c1.py` with the
   actual observed effect size instead of the hand-picked `b_gap=2.5`;
   that single step would replace most of the speculation in both this
   document and the closure session's C6 work with a real number.
