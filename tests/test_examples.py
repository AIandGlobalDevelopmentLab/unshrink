from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_ate_workflow_example_runs():
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "ate_workflow.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Calibration recommendation:" in result.stdout
    assert "Debiased ATE:" in result.stdout
