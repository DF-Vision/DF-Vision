"""
c1_shap_check_SYNTHETIC.py -- SYNTHETIC - NOT A RESULT

C1 acceptance verification: on mock_data_generator.py's planted-effect
data (known b_gap=2.5, strong signal), confirm rf_shap_check() (pooled)
and rf_shap_check_per_object() rank the planted Gap feature in the top-2
by whichever method (shap or fallback) is active in this environment.
Writes a SYNTHETIC-stamped text table. See project_closure/CRITERIA.md's
C1 section for the pre-committed acceptance bar this script checks.
"""
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "perception-calibration-gap" / "mock-pipeline"))

import analysis_c1 as c1
from mock_data_generator import generate_mock_trials, TRUE_COEFS

OUT_PATH = Path(__file__).parent / "c1_shap_check_SYNTHETIC_output.txt"


def main():
    lines = []
    lines.append("SYNTHETIC - NOT A RESULT")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"shap available in this environment: {c1._SHAP_AVAILABLE}")
    if not c1._SHAP_AVAILABLE:
        lines.append(f"  (import error: {c1._SHAP_IMPORT_ERROR})")
    lines.append(f"Ground truth TRUE_COEFS (mock_data_generator.py): {TRUE_COEFS}")
    lines.append("")

    # n_per_object=300: see tests/test_rf_shap_check.py's planted_df fixture
    # docstring -- real SHAP's pooled ranking is sample-size-limited at
    # mock_data_generator's DEFAULT_N_PER_OBJECT=30 (N=90 total).
    df = generate_mock_trials(
        n_per_object={"apple": 300, "purse": 300, "glass": 300},
        seed=20260921, gap_signal_multiplier=1.0,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pooled = c1.rf_shap_check(df)
        for w in caught:
            lines.append(f"[WARNING] {w.message}")

    lines.append(f"\n=== POOLED (importance_method={pooled['importance_method']}) ===")
    lines.append(f"rf_cv_auc_mean={pooled['rf_cv_auc_mean']:.3f} rf_cv_auc_std={pooled['rf_cv_auc_std']:.3f}")
    lines.append(f"top2_features={pooled['top2_features']}")
    lines.append(str(pooled["shap_importance_ranked"]))
    pooled_gap_top2 = any("Gap" in f for f in pooled["top2_features"])
    lines.append(f"Gap in top-2 (pooled): {'PASS' if pooled_gap_top2 else 'FAIL'}")

    per_obj = c1.rf_shap_check_per_object(df)
    lines.append(f"\n=== PER-OBJECT (importance_method varies per object) ===")
    per_obj_pass = {}
    for obj, result in per_obj.items():
        if result["top2_features"] is None:
            lines.append(f"  {obj}: {result['note']}")
            continue
        gap_top2 = any("Gap" in f for f in result["top2_features"])
        per_obj_pass[obj] = gap_top2
        lines.append(f"  {obj}: method={result['importance_method']} "
                      f"top2={result['top2_features']} Gap_in_top2={'PASS' if gap_top2 else 'FAIL'}")

    n_evaluable = len(per_obj_pass)
    n_pass = sum(per_obj_pass.values())
    lines.append(f"\nPer-object: {n_pass}/{n_evaluable} objects show Gap in top-2 "
                  f"(evaluable objects only, insufficient-data objects excluded from this ratio)")

    overall = pooled_gap_top2 and n_evaluable > 0 and n_pass >= 1
    lines.append(f"\n=== OVERALL: {'PASS' if overall else 'FAIL'} "
                  f"(pooled top-2 AND at least one evaluable object's top-2) ===")

    text = "\n".join(lines)
    print(text)
    OUT_PATH.write_text(text + "\n")
    print(f"\nSYNTHETIC output written: {OUT_PATH}")
    return overall


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
