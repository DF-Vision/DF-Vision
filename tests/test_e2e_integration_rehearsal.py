"""
tests/test_e2e_integration_rehearsal.py

C8: keeps perception-calibration-gap/e2e_integration_rehearsal_SYNTHETIC.py
green, per the task's "add it ... and keep that check green" instruction.
Runs the actual script as a subprocess (matches how a human/CI would
invoke it) and asserts exit code 0 plus the key SYNTHETIC-stamping and
assertion markers appear in its output.
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "perception-calibration-gap" / "e2e_integration_rehearsal_SYNTHETIC.py"


def test_e2e_rehearsal_script_exits_zero_and_passes_all_stages():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    out = result.stdout
    assert "SYNTHETIC - NOT A RESULT" in out
    assert "E2E INTEGRATION REHEARSAL: PASS" in out
    assert "prg_is_stub=True" in out
    assert "Motor interface call count: 0 (must be 0)" in out
    assert "Freeze check correctly FAILS on the deliberate change" in out


def test_e2e_rehearsal_leaves_no_stray_files_in_repo_root():
    """The script uses a tempfile.TemporaryDirectory for everything it
    writes -- confirms no e2e_SYNTHETIC_* files leak into the repo root
    (a real risk this task's own review caught once already, fixed by
    moving the freeze-check log paths into the tmp dir)."""
    before = set(PROJECT_ROOT.glob("e2e_SYNTHETIC*"))
    subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
    )
    after = set(PROJECT_ROOT.glob("e2e_SYNTHETIC*"))
    assert before == after, f"stray files leaked into repo root: {after - before}"
