# Calibration Gap × PRG — How to Run Everything

This file was empty before `project/closure` (C9). One command per
script/figure/table this repo can currently produce, grouped by stage.
**Every command below that touches `mock-pipeline`, `explore`, or
`project_closure` data is SYNTHETIC — never real trial evidence.** Real
trials require the physical rig; see `project_overnight/HARDWARE_CHECKLIST.md`.

Run everything from the repo root unless noted. Most scripts need
`sys.path` entries they set up themselves; just `python3 <path>`.
**Two exceptions, confirmed this session** (both use a path relative to
CWD, not to the script's own file, and DO fail from the repo root):
`mock_data_generator.py` and `hand_calc_validation.py` both need
`cd perception-calibration-gap` first — commands below are written that
way for those two specifically.

## 0. Tests (run this first)

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -v
```

## 1. Wiring/smoke checks

```bash
python3 perception-calibration-gap/daily_integration_check.py       # synthetic wiring check, real Gap/occlusion/TSDF modules + stub PRG
python3 perception-calibration-gap/e2e_integration_rehearsal_SYNTHETIC.py   # C8: full chain, replay->Gap->calibration->lookup-table->logger->stats->conformal
```

## 2. Real-camera capture (needs hardware) / replay (no hardware)

```bash
python3 perception-calibration-gap/real_camera/generate_replay_fixture.py   # writes a SYNTHETIC fixture set (4 good + 4 deliberately faulted trials)
python3 perception-calibration-gap/real_camera/capture_and_gap.py --replay perception-calibration-gap/real_camera/fixtures/replay_synthetic_v1
python3 perception-calibration-gap/real_camera/capture_and_gap.py            # LIVE mode -- needs Orbbec Gemini 2 + printed markers, see HARDWARE_CHECKLIST.md
```

## 3. Calibration (T1) and grasp lookup table (T2) — infrastructure, dry-run only

```bash
python3 perception-calibration-gap/calibration/compute_extrinsic_calibration.py --help   # T1: camera-to-arm extrinsic (Kabsch/SVD), needs real point_pairs.csv
python3 perception-calibration-gap/grasp/lookup_table_executor.py --help                  # T2: dry-run only, never pass --i-am-supervising without a human present
```

## 4. Checkpoint 1 (ground-truth marker validation)

```bash
cd perception-calibration-gap && python3 hand_calc_validation.py && cd ..   # CWD-relative path; exits 1 by DESIGN (refuses the still-template validation_scenes.csv) -- not a broken command
```

## 5. YOLO (T5)

```bash
python3 -c "import sys; sys.path.insert(0,'perception-calibration-gap/yolo'); import train_yolo_scaffold as t; t.run_sanity_finetune(epochs=2, imgsz=320)"   # real 2-epoch training, CODE CHECK only, SYNTHETIC boxes
python3 -c "import sys; sys.path.insert(0,'perception-calibration-gap/yolo'); import per_object_gate as g; g.print_report(g.evaluate_gate(g.demo_metrics))"   # per-object acceptance gate demo
```

## 6. Gap-only statistics (C1)

```bash
(cd perception-calibration-gap && python3 mock-pipeline/mock_data_generator.py)   # writes mock-pipeline/data/mock_trials.csv (SYNTHETIC, CWD-relative path)
python3 perception-calibration-gap/analysis/analysis_c1.py                    # Models A/B/C, VIF, bootstrap delta-AUC, RF+SHAP(-or-fallback), per-object table
python3 project_closure/scripts/c1_shap_check_SYNTHETIC.py                    # SHAP/fallback acceptance check on planted data (C1)
python3 perception-calibration-gap/analysis/power_analysis_c1.py --n-reps 300 # power curve, multiple N/effect-size cells
python3 project_closure/scripts/c6_statistics_SYNTHETIC.py                    # C6: null FPR (1000 reps), planted recovery (500x3), power at N=90/120/180, conformal spread
```

## 7. Two-signal path (C4/C4-ext) — conditional, guarded off by default

```bash
python3 perception-calibration-gap/analysis/analysis_c4.py     # Gap conformal gate + FAIL-Detect baseline + ablation + KS check, SYNTHETIC
python3 perception-calibration-gap/analysis/model_d.py         # refuses unless analysis/model_d_config.json {"enabled": true}
python3 perception-calibration-gap/analysis/power_analysis_c3.py
python3 perception-calibration-gap/analysis/analysis_c2.py
python3 perception-calibration-gap/analysis/analysis_c3.py
python3 perception-calibration-gap/analysis/analysis_c5.py
```

## 8. PRG signal itself — classifier (default) vs classifier-free candidates (off by default, negative result)

```bash
python3 explore/scripts/e3_evaluate_candidates.py    # re-evaluates C1/C2/C3 against explore/CRITERIA.md -- decisive negative result, see explore/E3_FINDINGS.md and project_closure/LOG.md's C4 entry for the post-fix re-run
```

## 9. Trial harness / session planning / data quality

```bash
python3 perception-calibration-gap/analysis/session_scheduler.py         # rotated object-order schedule, demo
python3 perception-calibration-gap/daily_integration_check.py            # writes a demo log this next command can read
python3 perception-calibration-gap/analysis/data_quality_report.py --log perception-calibration-gap/mock-pipeline/data/trials_daily_integration.jsonl
```

## 10. Mutation testing (C7) — proves the guards' tests can actually fail

```bash
python3 project_closure/scripts/c7_run_mutations.py   # 30 mutations across 10 guarded modules, ~1-2 min
```

## 11. Verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -v
find . -iname "*.joblib" | sort | xargs sha256sum   # compare against project_closure/LOG.md's C0/C10 baselines
git diff <branch-point>..HEAD --stat -- '*.joblib' integration/prg_config.json perception-calibration-gap/PREREG.md "SCHEMA.md " "INTERFACE.md "   # must be empty
```

See `REPRODUCE.md` (repo root) for a shorter, output-oriented checklist
mapping specific figures/tables to the commands above.
