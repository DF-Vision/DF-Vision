#!/usr/bin/env python3
"""
explore/scripts/check_structural_independence.py

Extends the structural-independence audit referenced in PROJECT_STATUS.md
section 2 ("AST-parsed every import in extract_relation_features.py,
classify_relation.py, compute_prg_edge_error.py, build_prg.py, plus every
Person-A file... zero internal imports... zero overlap") -- that
original audit's own script was never committed to this repo (confirmed
via repo-wide search across three prior sessions), so this is a fresh,
from-scratch AST audit covering the SAME PRG-side files plus the two new
ones this session added: explore/scripts/geometric_prg_error.py and the
updated compute_prg_edge_error.py.

Checks, via Python's ast module (not just grep, so string literals /
comments mentioning "yolo" etc. don't produce false positives):
  1. No import statement in the new geometric module or the updated
     compute_prg_edge_error.py references anything under
     perception-calibration-gap/ (Person A's YOLO/TSDF/Gap/occlusion/
     conformal code) or any detector-confidence-bearing module.
  2. Transitive closure: every module the PRG-side files import,
     recursively, stays within relation_classifier/, explore/, or the
     standard library / third-party packages (numpy, cv2, scipy,
     sklearn, joblib) -- never crosses into Person A's directory tree.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRG_SIDE_FILES = [
    PROJECT_ROOT / "relation_classifier" / "scripts" / "extract_relation_features.py",
    PROJECT_ROOT / "relation_classifier" / "scripts" / "classify_relation.py",
    PROJECT_ROOT / "relation_classifier" / "scripts" / "compute_prg_edge_error.py",
    PROJECT_ROOT / "relation_classifier" / "scripts" / "build_prg.py",
    PROJECT_ROOT / "explore" / "scripts" / "geometric_prg_error.py",
]

FORBIDDEN_PATH_MARKERS = ["perception-calibration-gap", "yolo", "tsdf_fusion",
                           "gap_computer", "conformal_gate", "occlusion_estimator"]


def get_imports(py_file: Path) -> list[str]:
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


def resolve_local_module(module_name: str, from_file: Path) -> Path | None:
    """Best-effort: try to find a local .py file for a dotted module name,
    searching relative to the project root and the importing file's own
    directory (covers both absolute 'relation_classifier.scripts.x' style
    and the sys.path.insert-then-bare-import style this codebase actually
    uses throughout)."""
    candidates = [
        PROJECT_ROOT / (module_name.replace(".", "/") + ".py"),
        from_file.parent / (module_name.split(".")[-1] + ".py"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    print("=" * 78)
    print("Structural independence audit (AST-based) -- PRG side vs Person A's files")
    print("=" * 78)

    all_clean = True
    visited: set[Path] = set()
    to_visit: list[Path] = list(PRG_SIDE_FILES)

    while to_visit:
        f = to_visit.pop()
        if f in visited or not f.exists():
            continue
        visited.add(f)

        imports = get_imports(f)
        print(f"\n{f.relative_to(PROJECT_ROOT)}:")
        print(f"  imports: {imports}")

        for name in imports:
            lname = name.lower()
            if any(marker in lname for marker in FORBIDDEN_PATH_MARKERS):
                print(f"  [FAIL] import {name!r} matches a forbidden Person-A marker")
                all_clean = False
                continue
            local = resolve_local_module(name, f)
            if local is not None:
                if any(marker in str(local).lower() for marker in FORBIDDEN_PATH_MARKERS):
                    print(f"  [FAIL] import {name!r} resolves to Person-A path {local}")
                    all_clean = False
                else:
                    to_visit.append(local)

    print(f"\n{'='*78}")
    print(f"Files audited (transitive closure): {len(visited)}")
    for f in sorted(visited):
        print(f"  {f.relative_to(PROJECT_ROOT)}")
    print(f"\nRESULT: {'PASS -- zero Person-A imports found, in this file set or its transitive closure' if all_clean else 'FAIL -- see above'}")
    sys.exit(0 if all_clean else 1)


if __name__ == "__main__":
    main()
