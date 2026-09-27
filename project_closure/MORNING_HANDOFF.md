# Morning Handoff — `project/closure`

Autonomous overnight CLOSURE session, 2026-09-21. Branch `project/closure`,
created from `project/overnight-hardening` @ `6864339`. Not merged, not
pushed. Purpose: finish everything the prior overnight session deferred
or flagged, and prove the existing guards actually work.

## 1. Completed ledger (see `project_closure/LEDGER.md` for full detail)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | T4 `--replay` mode | **CLOSED** | `perception-calibration-gap/real_camera/`, `tests/test_replay_mode.py` 14/14 |
| 2 | `rf_shap_check()` / broken shap | **CLOSED** | `analysis_c1.py` guarded import + fallback, `tests/test_rf_shap_check.py` 8/9 (+1 skip, verified in isolated venv) |
| 3 | YOLO pipeline verification | **CLOSED** | `perception-calibration-gap/yolo/`, `tests/test_yolo_pipeline.py` 15/15, `tests/test_yolo_real_training.py` 4/4 |
| 4 | C1 gap estimator returns 0.0 | **CLOSED** | `explore/scripts/geometric_prg_error.py`, `tests/test_geometric_prg_error_sanity.py`, `tests/test_c4_gap_estimator_scenes.py` — fixed, but see §3 verdict below |
| 5 | T6b/T6c changed-interface PRG attempt | **See below** — attempted after this handoff was drafted; check `project_closure/LEDGER.md` row 5 for the final status word (CLOSED or BLOCKED) since this file may have been written before or after that attempt concluded |
| 6 | Null FPR re-establishment | **CLOSED** | `project_closure/scripts/c6_statistics_SYNTHETIC_output.txt`, 1000/500×3/100+ repeats |
| 7 | `hand_calc_validation.py` positive-path test | **CLOSED** (already existed from a prior session, confirmed via mutation, not rebuilt) | `tests/test_hand_calc_validation.py::test_passes_once_notes_are_filled_in_for_real` |
| 8 | Other-branch commits, human merge decision needed | **CLOSED** (left untouched, as instructed) | `git branch -a` unchanged |

Plus 6 additional grep-sourced ledger rows (9-14 in `LEDGER.md`), all
CLOSED as either intentional/already-tested guards or confirmed out of
this session's scope.

## 2. Task table

| Task | Status | Evidence |
|---|---|---|
| C0 Baseline + ledger | DONE | `project_closure/LOG.md` C0 entry |
| C2 `--replay` mode | DONE | LOG C2 entry, 14 tests |
| C1 shap fix/fallback | DONE | LOG C1 entry, 9 tests (8 pass + 1 env-conditional skip) |
| C4 Gap estimator fix + re-run | DONE (fix verified; E3 negative result unchanged) | LOG C4 entry, 6 tests + `c4_e3_rerun_SIM_output.txt` |
| C3 YOLO verification | DONE | LOG C3 entry, 19 tests |
| C7 Mutation testing | DONE | LOG C7 entry, 30/30 kill rate (after closing 1 gap) |
| C6 Statistics re-establishment | DONE | LOG C6 entry, 12 tests + `c6_statistics_SYNTHETIC_output.txt` |
| C8 E2E rehearsal | DONE | LOG C8 entry, 2 tests |
| C9 Schema/doc closure | DONE | LOG C9 entry, 5 doc files |
| C10 Verification | DONE | LOG C10 entry |
| C11 Handoff | DONE | this file |
| C5 PRG changed-interface attempt | See `project_closure/LEDGER.md` row 5 / `LOG.md`'s C5 entry if present — lowest priority per the task's own STOP CONDITIONS ordering, attempted last, budget permitting |

## 3. C4 verdict, in three sentences

The C4 fix genuinely resolves the estimator's degenerate 0.0-on-clean-edge
bug — confirmed on 4 new idealized/synthetic unit scenes (single-edge,
two-object-touching, separated-with-real-gap, background-only), all now
passing exactly. The earlier decisive negative result on real,
multi-object, cluttered simulation scenes (`explore/E3_FINDINGS.md`)
does NOT change and, on the validity metric (AUROC), is measurably
worse (C1's AUROC drops from 0.457 to 0.280, further below chance) —
sensitivity remains exactly 0/5 worlds for all three candidates,
identical to before. This confirms the zero-return bug and the
E3-diagnosed omnidirectional background-contamination mechanism are
DISTINCT root causes, and fixing the first did not fix (and by one
metric, worsened) the second.

## 4. Mutation kill-rate table

| Target | Mutations | Result |
|---|---|---|
| 1. Calibration solver + refusal | 3 | 3/3 killed |
| 2. Lookup-table/out-of-reach | 3 | 2/3 killed first attempt, 1 survivor closed (joint-limit boundary), then 3/3 |
| 3. Dry-run/`--i-am-supervising` | 3 | 3/3 killed |
| 4. Marker-size config read | 3 | 3/3 killed |
| 5. Template-CSV guard | 3 | 3/3 killed |
| 6. Model D stub-exclusion | 3 | 3/3 killed |
| 7. Session scheduler | 3 | 3/3 killed |
| 8. `data_quality_report` | 3 | 3/3 killed |
| 9. Controller-freeze check | 3 | 3/3 killed |
| 10. Depth-unit guard | 3 | 3/3 killed |
| **Total** | **30** | **30/30 (100%) after closing 1 test gap** |

A harness bug (stale bytecode cache risking false "killed" results) was
found and fixed mid-task; all 30 mutations were re-run clean after the
fix — the table above is the post-fix result.

## 5. C6 power conclusions, in plain words

**The C1 study, as currently planned (N≈90-180 trials), cannot reliably
detect the Gap coefficient's significance at the effect size the
project's own mock generator assumes is realistic — not even close.**
Power at N=90 is 8%, at N=180 it's 11% (both far below the conventional
80% target). Even inflating N sixteen-fold, to 1440 trials — a number
far beyond what real data collection could plausibly reach — power only
gets to 59%, still short. This was double-checked to rule out a code
bug: artificially cranking up the assumed effect size (not the real
one) does drive power to 100%, so the statistics code itself works
correctly — the study design, at realistic assumptions, is just
underpowered. On the bright side, the model's confidence intervals are
honest (95-97% actual coverage against a 95% target), and the null
false-positive rate is well-calibrated (4.4%, essentially the expected
5%) — so nothing here suggests the code lies, only that the planned
sample size can't support a significance claim at the assumed effect
size. **This needs your decision**: collect far more real trials than
planned, use a less strict test than significance (e.g., report the
effect estimate with an honest wide CI), or otherwise reframe what C1's
paper claim can be.

## 6. Every UNVERIFIED item

- **Everything about real hardware** — no camera, no arm, no real trial
  ever collected this session (or any prior session). Checkpoint 1 is
  still refused (template `validation_scenes.csv`).
- **DA-V2/real-depth behavior of any PRG signal** — still entirely
  untested; `rclpy` still absent, DA-V2 checkpoints exist only outside
  the repo (`/home/manasi/bps-project/checkpoints/`), not copied in or
  used.
- **Real SHAP verification** was done in an isolated, throwaway
  project-local venv (not the default interpreter) — confirms the
  fallback logic is sound and the real path works when available, but
  the DEFAULT interpreter here still runs the fallback, not real SHAP.
- **C5 (T6b/T6c changed-interface PRG attempt)** — see `LEDGER.md` row 5
  for final disposition; may not have been reached if budget ran out
  first, per the task's explicit lowest-priority ordering.
- **The mock generator's assumed effect size itself** (`b_gap=2.5`) is
  a modeling choice, not measured from real data — C6's underpowering
  finding is conditional on that assumption being in the right
  ballpark; if the real effect turns out much larger, the study could
  still be adequately powered. This has never been checked against
  anything real.
- **`analysis_c2.py`, `analysis_c3.py`, `analysis_c5.py`, `power_analysis_c3.py`**
  were run once each (README verification, C9) but not otherwise
  audited/tested this session — their own internal pass/fail logic was
  taken at face value.

## 7. What software CANNOT tell you until real data exists

- **Calibration residual on real measured points** — `compute_extrinsic_calibration.py`
  is verified against noiseless synthetic transforms only; real marker
  detection noise, lens distortion, and hand-measurement error are
  untested.
- **Marker-size and frame-convention correctness on the physical rig** —
  every marker size in `manifest.csv` is a stated value, never
  ruler-verified against an actually-printed marker.
- **YOLO detector quality** — every training run this session used
  solid-color synthetic boxes; zero real captured/labeled images exist.
- **Real-depth behavior of any PRG signal** (classifier-based or
  geometric) — all evaluation has been on ray-cast/simulated depth.
- **Failure prediction itself** — Gap, PRG, and the conformal gate have
  never been tested against a single real grasp attempt's outcome.

## 8. Exact review commands

```bash
cat project_closure/LEDGER.md                 # full ledger, evidence pointers
cat project_closure/CRITERIA.md               # every pre-committed threshold, in commit order
cat project_closure/LOG.md                    # full command/output trail, every task
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -v
git log --oneline project/overnight-hardening..HEAD
git diff project/overnight-hardening..HEAD --stat -- '*.joblib' integration/prg_config.json perception-calibration-gap/PREREG.md "SCHEMA.md " "INTERFACE.md "
python3 project_closure/scripts/c7_run_mutations.py     # re-run mutation testing (~1-2 min)
python3 project_closure/scripts/c6_statistics_SYNTHETIC.py   # re-run full-scale statistics (~90s)
python3 perception-calibration-gap/e2e_integration_rehearsal_SYNTHETIC.py
```

## 9. Recommended first action

Two independent tracks, neither blocking the other:

1. **Hardware** (per `project_overnight/HARDWARE_CHECKLIST.md`, unchanged
   by this session): print + measure markers, run Checkpoint 1. Nothing
   this session did changes that recommendation.
2. **Statistical design decision** (new, urgent, from C6): read
   `PREREG_AMENDMENTS_PROPOSED.md` item 7 (the power-analysis finding)
   before collecting a single real trial — if C1's planned N can't
   support the significance claim at the assumed effect size, that's
   worth knowing BEFORE running the study, not after.

## 10. Claims-versus-evidence checklist

| Claim | Evidence |
|---|---|
| `--replay` mode runs the identical per-trial code path as live capture, no camera needed | `tests/test_replay_mode.py`, 14/14 passing |
| Depth-unit conversion (mm PNG → m) happens in exactly one place, with a plausibility guard | `depth_io.py`, `tests/test_replay_mode.py`'s conversion/guard tests |
| `rf_shap_check()` works (real SHAP or a labeled fallback) and ranks the planted Gap feature top-2, pooled and per-object | `tests/test_rf_shap_check.py`, `c1_shap_check_SYNTHETIC_output.txt` |
| C1's gap estimator recovers the true gap on 4 idealized synthetic scenes | `tests/test_geometric_prg_error_sanity.py`, `tests/test_c4_gap_estimator_scenes.py` |
| Re-running `explore/CRITERIA.md` unchanged after the C4 fix still shows a decisive negative result on real scenes | `project_closure/results/c4_e3_rerun_SIM_output.txt` |
| YOLO training loop mechanics work (loss decreases, checkpoints save/reload/predict) | `tests/test_yolo_real_training.py`, real 2-epoch run |
| Auto-labeler recovers known boxes within 1px; class index is a single source of truth; train/val split is session-grouped | `tests/test_yolo_pipeline.py` |
| All 10 named guards have mutation-tested, non-vacuous test coverage | `project_closure/LOG.md`'s C7 entry, 30/30 kill rate |
| Null false-positive rate is well-calibrated at N=90 (1000-dataset exact binomial CI) | `c6_statistics_SYNTHETIC_output.txt` |
| C1's planned study is severely underpowered at the assumed effect size, even at 16x the planned N | `c6_statistics_SYNTHETIC_output.txt`, verified not to be a code bug |
| Conformal coverage's single-seed anomaly from a prior session was noise, not a systematic issue | `c6_statistics_SYNTHETIC_output.txt`, 100-split spread |
| FAIL-Detect baseline is never trained on held-out test rows | `tests/test_split_discipline.py`, fixed a previously-vacuous assertion in `analysis_c4.py` |
| E2E rehearsal chains 7 real stages (replay→Gap→calibration→lookup-table→logger→data-quality→stats→conformal) with zero motor calls | `tests/test_e2e_integration_rehearsal.py`, `e2e_integration_rehearsal_SYNTHETIC.py` |
| All protected files (`.joblib`, `prg_config.json`, `PREREG.md`, `SCHEMA.md `, `INTERFACE.md `) byte-identical throughout | SHA256 diff, C0 vs C10, `project_closure/LOG.md` |
| Fresh-clone reproduction confirmed | `project_closure/LOG.md`'s C10 entry |
| No arm/motor command was ever sent; `--i-am-supervising` never passed | Every grasp-facing test asserts the mock `MotorInterface` is called zero times; `git grep -- --i-am-supervising` shows it's never passed by any script this session ran |

---

## Draft team update (NOT sent)

Closure pass on top of last night's hardening session — finished
everything that was flagged or deferred, and spent real effort proving
the existing safety/statistics guards actually work rather than just
existing.

Fixed the real-camera pipeline's `--replay` mode (found along the way
that it had literally never been importable in this environment before —
a broken path setup, not just untested), the broken `shap`/SHAP-importance
path (with a working fallback plus a verified real-SHAP path in an
isolated venv), and the YOLO pipeline's missing pieces (auto-labeler,
class-index consistency, session-grouped splitting, a real 2-epoch
training run that actually decreases loss).

Traced and fixed the root cause of the geometric gap estimator returning
exactly 0.0 on clean synthetic edges — but re-running the pre-committed
evaluation criteria shows this does NOT rescue the PRG signal on real
scenes; if anything it's slightly worse by one metric. The classifier-free
PRG path remains a decisive negative result across five sessions now.

Mutation-tested all 10 named safety/statistics guards (30 injected bugs,
100% caught after fixing one real test gap) — this is now real evidence
the test suite can actually catch a broken guard, not just that tests
exist.

The single most important finding: re-ran the null/power statistics at
the scale you asked for (1000+ datasets instead of ~50-100) and found the
planned study is severely underpowered at the assumed effect size — even
at 16x the planned sample size. This needs a decision before real data
collection starts: bigger N, a different statistical framing, or both.
Flagged as a dated PREREG amendment for your sign-off, not silently
acted on.

Nothing here is real-hardware evidence. `use_stub` stays `true`
throughout; every `.joblib`/`PREREG.md`/`SCHEMA.md `/`INTERFACE.md ` file
is untouched (SHA256-verified); no arm or motor command was ever issued.
