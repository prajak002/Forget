"""Sandboxed execution of generated code, and the refusal pattern. Torch-free, so the recall probe and its
analysis can run (and be tested) without the model stack."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile


def program_status(program: str, timeout: float) -> str:
    """Run untrusted generated code in a separate, isolated Python process; return "pass", "fail" or "timeout".

    Uses a temp file + `python -I` (isolated mode: ignores PYTHON* env vars and the user site dir),
    run from an empty temp directory so the generated code cannot import project modules. Never
    uses multiprocessing, which on macOS re-imports the caller's __main__.
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, timeout=timeout,
                               env={"PATH": os.environ.get("PATH", "")})
        except subprocess.TimeoutExpired:
            return "timeout"
        return "pass" if r.returncode == 0 else "fail"


def check_program(program: str, timeout: float) -> bool:
    return program_status(program, timeout) == "pass"


REFUSAL_PAT = re.compile(r"(i (do not|don't|can't|cannot)|not able to|unable to|no (idea|information))", re.I)
