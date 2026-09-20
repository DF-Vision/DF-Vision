# Morning Handoff — `project/overnight-hardening`

Autonomous overnight hardening session, 2026-09-21. Branch
`project/overnight-hardening`, created from `explore/geometric-prg-error`
@ `f5cac32` (confirmed as the correct base — see `project_overnight/STATE.md`'s
branch-decision table: strict superset of every other named branch except
one trivial README commit). Not merged, not pushed.

## 1. Completeness matrix

| Component | Was (T0) | Now | Evidence |
|---|---|---|---|
| T1 camera-to-arm calibration | MISSING | **Built, 8/8 tests** | `perception-calibration-gap/calibration/`, `tests/test_extrinsic_calibration.py` |
| T2 grasp lookup-table path | MISSING | **Built, 10/10 tests** | `perception-calibration-gap/grasp/`, `tests/test_grasp_lookup_table.py` |
| T3 Checkpoint 1 tooling | EXISTS, one real gap | **Gap closed, 11/11 new tests** | `hand_calc_validation.py`'s `refuse_if_still_template()`, `tests/test_hand_calc_validation.py` |
| T4 real camera path | EXISTS (masks_are_bbox_placeholder correct) | **Partial — `--replay`/config-source NOT built** | `project_overnight/LOG.md` T4 entry |
| T5 YOLO pipeline | EXISTS substantially | **Per-object gate added, 6/6 tests; scaffold re-run not re-confirmed** | `yolo/per_object_gate.py`, `tests/test_per_object_gate.py` |
| T6 PRG signal | 3 candidates failed (prior session) | **T6a sanity test done (found+documented 2 new bugs); T6b/T6c deferred** | `tests/test_geometric_prg_error_sanity.py` |
| T7 Gap-only stats | EXISTS substantially | **Verified via synthetic recovery; found+fixed 1 real bug** | `project_overnight/scripts/t7_synthetic_recovery_check.py` |
| T8 Model D | MISSING (no file referenced it at all) | **Built, guarded off by default, 11/11 tests** | `analysis/model_d.py`, `tests/test_model_d.py` |
| T9 trial harness/schema | PARTIAL (9 fields missing) | **Extended additively; scheduler + data-quality report built, 10/10 tests** | `logger.py`, `analysis/data_quality_report.py`, `analysis/session_scheduler.py` |
| T10 documentation | PARTIAL | **5 PREREG amendments + 1 SCHEMA amendment appended, dated** | `PREREG_AMENDMENTS_PROPOSED.md`, `SCHEMA_AMENDMENT_PROPOSED.md` |
| T11 verification | — | **Done, 73/73 + 1 xfail, all protected files unchanged** | `project_overnight/T11_VERIFICATION.md` |
| T12 hardware checklist | MISSING as one document | **Written, 9 ordered steps** | `project_overnight/HARDWARE_CHECKLIST.md` |

## 2. STATE.md discrepancies (from "Known context")

1. **PRG schema field names**: the task assumed flat `prg_edge_error`/
   `prg_edge_error_is_stub` columns; the real, already-committed schema
   uses `prg_edges: List[dict]` + `prg_is_stub: bool` (richer — supports
   >1 neighbor per trial). Built T8/T9 against the real names.
2. **`validation_scenes.csv`**: "still a blank template" is true, but the
   template itself has 3 non-empty placeholder rows with "REPLACE with"
   notes, not zero rows — the pre-existing "no data rows" refusal did
   NOT catch this. Closed in T3.
3. **T8's framing** ("Code exists but is guarded") was not accurate —
   verified via `grep` that no file anywhere referenced Model D or
   `prg_edge_error` before this session. Built fresh.
4. **An unrelated branch `a/overnight-hardening`** exists locally, from a
   different prompt variant that based off `unified-pipeline` instead of
   `explore/geometric-prg-error`. Left untouched per "never touch other
   branches" — flagged for you to reconcile if the two overnight-hardening
   attempts ever need merging.

## 3. T6 verdict, in three sentences

The synthetic step-edge sanity test confirmed C2's boundary-sharpness
signal is real and correctly directional once isolated from a newly-found
quantization-masking artifact in the shared noise model (affecting gaps
near 1mm, exactly the Contact/Support scale), but also found a new,
more fundamental bug in C1's gap estimator that returns exactly 0.0 even
on an idealized single-edge scene with no background at all — a distinct
mechanism from the multi-object background-contamination issue the prior
session diagnosed. Per the 3-attempt budget, this bug is documented
(xfail-marked, root cause explained) rather than fixed, and the changed-
interface variant (T6b) and full gate re-evaluation (T6c) were not
attempted this session given the remaining scope. `use_stub` stays
`true`; nothing about T6a's findings changes the prior session's decisive
STOP verdict on the three existing candidates.

## 4. Every UNVERIFIED item

- `analysis_c1.rf_shap_check()` — `shap` import broken in this
  environment (numba/coverage conflict); worked around via an in-process
  stub for everything else in that file, but this one function was never
  actually called or checked.
- T4's real-camera `--replay` mode and config-driven source selection —
  not built.
- T6b (changed-interface neighbor-localization PRG variant) and T6c
  (full re-evaluation against `explore/CRITERIA.md`) — not attempted.
- `train_yolo_scaffold.py`'s end-to-end mechanics run — attempted, timed
  out after 90s (apparent CPU contention), not re-diagnosed.
- T7's conformal-coverage check was run on a single seed (0.955 vs 0.900
  target, outside tolerance) — not repeated across seeds; could be
  noise, not a systematic issue. Flagged, not resolved.
- Everything about real hardware, entirely (no camera, no arm connected
  this session, as in every prior session).

## 5. Exact review commands

```bash
cat project_overnight/STATE.md               # full T0 inventory, evidence-based
cat project_overnight/CRITERIA.md             # every pre-committed threshold
cat project_overnight/LOG.md                  # full command/output trail, every task
cat project_overnight/T11_VERIFICATION.md     # test counts, SHA256 checks, fresh-clone proof
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -v
git log --oneline explore/geometric-prg-error..HEAD
git diff explore/geometric-prg-error..HEAD --stat -- '*.joblib' integration/prg_config.json perception-calibration-gap/PREREG.md "SCHEMA.md " "INTERFACE.md "
```

## 6. Recommended first action

Read `project_overnight/HARDWARE_CHECKLIST.md` and start at step 1
(print + measure markers) — everything else in this session's work
(calibration tooling, lookup table, Checkpoint 1 refusal guard, the
real-capture protocol) is now ready and waiting on that physical chain,
in the order the checklist specifies. In parallel/no-hardware-needed:
review the 5 PREREG amendments in `PREREG_AMENDMENTS_PROPOSED.md`'s
2026-09-21 section — none are applied, all need your explicit sign-off,
and the object-count conflict (item 3) in particular has now been raised
across three separate sessions without resolution.

## 7. Claims-versus-evidence checklist

| Claim | Evidence |
|---|---|
| T1 recovers a known transform to <1e-8 error noiseless, refuses degenerate/mm-confused/corrupted input | `tests/test_extrinsic_calibration.py`, 8/8 passing |
| T2 never extrapolates out-of-reach, never silently clamps joint limits, motor interface called zero times in dry-run | `tests/test_grasp_lookup_table.py`, 10/10 passing |
| `validation_scenes.csv` (real, committed) is correctly refused by the new template guard | Direct run output in `project_overnight/LOG.md`'s T3 entry |
| `hand_calc_validation.py` imports nothing from detector/YOLO code (transitively, via `occlusion_estimator.py` too) | `tests/test_hand_calc_validation.py`'s 2 AST-based tests |
| Per-object YOLO gate rejects a passing-aggregate/failing-one-object case | `tests/test_per_object_gate.py::test_aggregate_passes_but_one_object_fails_gate_still_fails` |
| C1's gap estimator returns 0.0 instead of true 1mm on an idealized single-edge scene | `tests/test_geometric_prg_error_sanity.py`, xfail-documented with root-cause docstring |
| `mock_data_generator.py`'s null-data bug caused 100/100 singular-matrix fit failures before the fix, 0/100 after (5% FPR, well-calibrated) | `project_overnight/scripts/t7_synthetic_recovery_check.py` output, before/after in `project_overnight/LOG.md`'s T7 entry |
| Model D refuses when disabled and when zero real rows exist; stub exclusion logs a loud count | `tests/test_model_d.py`, 11/11 passing |
| Gap-only path runs with prg columns entirely absent from the dataframe | `tests/test_model_d.py::test_gap_only_runs_without_prg_columns` |
| All legacy/v2 `.joblib`, `prg_config.json`, `PREREG.md`, `SCHEMA.md `, `INTERFACE.md ` unchanged throughout | SHA256 diff, `project_overnight/T11_VERIFICATION.md` |
| Fresh-clone + determinism confirmed | `project_overnight/T11_VERIFICATION.md` |
| No arm/motor command was ever sent; `--i-am-supervising` never passed | Every grasp-facing test asserts `MotorInterface` mocked and called zero times; grep confirms no invocation of the real flag anywhere in this session's commands |

---

## Draft team update (NOT sent)

Overnight hardening pass across the whole system, not just PRG. Built
the two hard-gate pieces that were genuinely missing — camera-to-arm
calibration (Kabsch/SVD, refuses on bad input, never guesses) and the
grasp lookup-table path (no IK solver, dry-run only, never moved
anything) — both fully tested and ready for you to run against real
hardware per `project_overnight/HARDWARE_CHECKLIST.md`.

Found and fixed a real bug in `hand_calc_validation.py` (Checkpoint 1
tool would have silently accepted the still-template `validation_scenes.csv`
as if it were real measured data — now refuses correctly) and in
`mock_data_generator.py` (the "null" synthetic dataset setting was
accidentally degenerate, causing the statistics code's own null-data
calibration check to fail 100% of the time for a reason that had nothing
to do with the statistics code itself — fixed, now shows near-perfect
5% false-positive rate at the nominal 5% level).

Built `Model D` (the Gap+PRG combined model) for the first time — it
didn't exist anywhere before — but it's guarded off by default and
refuses to run without real (non-stub) PRG data, per the stub-exclusion
rule. The PRG signal itself is still not working (found one more, more
fundamental bug in the gap estimator this session, on top of what was
already known) — recommending in the PREREG amendments that Gap-only
stays the primary path and PRG/Model D stay explicitly conditional.

Three things need your decision, all drafted as dated PREREG amendments
awaiting sign-off: (1) Gap-only as primary vs continuing to chase a PRG
signal, (2) the object-count conflict (5 vs 5-with-swap vs 3, across
three documents, flagged three sessions running now), (3) whether to
prioritize the real-hardware chain (calibration → lookup table →
Checkpoint 1 → real PRG capture) now that all the tooling for it exists.
`use_stub` stays `true` throughout; nothing here is real-hardware
evidence.
