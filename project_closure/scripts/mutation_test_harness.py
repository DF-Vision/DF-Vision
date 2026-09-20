#!/usr/bin/env python3
"""
mutation_test_harness.py

C7: hand-rolled mutation-testing harness (mutmut/cosmic-ray not used --
see project_closure/CRITERIA.md's C7 section for why). For each
mutation: apply a single textual substitution to a target file, run the
associated pytest test file, record kill/survive, then `git checkout --`
the target file to revert exactly, and verify `git status` is clean
before moving on. Never leaves a mutation applied on exit (even on
exceptions -- the revert happens in a finally block).
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Mutation:
    module: str          # short name for the table, e.g. "calibration solver"
    target_file: str      # repo-relative path
    test_file: str        # repo-relative path (or space-separated for multiple)
    description: str      # what bug this injects
    old: str               # exact substring to replace (must be unique in the file)
    new: str


@dataclass
class MutationResult:
    mutation: Mutation
    killed: bool
    test_output_tail: str = ""


def _git(*args):
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)


def _assert_clean():
    r = _git("status", "--porcelain")
    if r.stdout.strip():
        raise RuntimeError(f"Working tree not clean before/after mutation:\n{r.stdout}")


def apply_mutation(m: Mutation) -> None:
    path = PROJECT_ROOT / m.target_file
    text = path.read_text()
    count = text.count(m.old)
    if count != 1:
        raise ValueError(
            f"{m.module}: expected exactly 1 occurrence of old_string in {m.target_file}, found {count}"
        )
    path.write_text(text.replace(m.old, m.new))


def revert_mutation(m: Mutation) -> None:
    r = _git("checkout", "--", m.target_file)
    if r.returncode != 0:
        raise RuntimeError(f"git checkout failed for {m.target_file}: {r.stderr}")


def run_tests(test_file: str) -> subprocess.CompletedProcess:
    test_files = test_file.split()
    # PYTHONDONTWRITEBYTECODE: repeated rapid mutate/revert/mutate cycles on
    # the SAME file path can otherwise leave a stale .pyc whose recorded
    # source mtime happens to match the reverted file's new mtime (fast
    # write-revert-write within one filesystem timestamp tick) -- causing
    # Python to load mutated bytecode for what git says is clean, unmutated
    # source. Confirmed as a real failure mode this session (project_closure/
    # LOG.md's C7 entry): disable bytecode caching entirely for mutation runs.
    return subprocess.run(
        [sys.executable, "-B", "-m", "pytest", *test_files, "-q"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
        env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PATH": "/usr/bin:/bin",
             "PYTHONDONTWRITEBYTECODE": "1"},
    )


def run_mutation(m: Mutation) -> MutationResult:
    _assert_clean()
    try:
        apply_mutation(m)
        result = run_tests(m.test_file)
        killed = result.returncode != 0
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-15:])
        return MutationResult(mutation=m, killed=killed, test_output_tail=tail)
    finally:
        revert_mutation(m)
        _assert_clean()


def run_all(mutations: list[Mutation]) -> list[MutationResult]:
    results = []
    for m in mutations:
        print(f"[{m.module}] {m.description} ... ", end="", flush=True)
        r = run_mutation(m)
        print("KILLED" if r.killed else "SURVIVED")
        results.append(r)
    return results


def print_table(results: list[MutationResult]) -> None:
    print("\n| Module | Mutation | Result |")
    print("|---|---|---|")
    for r in results:
        status = "KILLED" if r.killed else "**SURVIVED**"
        print(f"| {r.mutation.module} | {r.mutation.description} | {status} |")
    n_killed = sum(1 for r in results if r.killed)
    print(f"\nKill rate: {n_killed}/{len(results)}")
    survivors = [r for r in results if not r.killed]
    if survivors:
        print("\nSURVIVED (test gaps):")
        for r in survivors:
            print(f"  - [{r.mutation.module}] {r.mutation.description}")
