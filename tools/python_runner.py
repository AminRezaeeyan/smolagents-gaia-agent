import subprocess
import sys
from pathlib import Path

from smolagents import Tool

TIMEOUT_SECONDS = 120


class RunPythonFileTool(Tool):
    name = "run_python_file"
    description = (
        "Executes a local Python file attached to the question in a separate process and returns "
        "its stdout and stderr. Use read_file first to see the code."
    )
    inputs = {"path": {"type": "string", "description": "Path of the local .py file."}}
    output_type = "string"

    def forward(self, path: str) -> str:
        file = Path(path).resolve()
        if not file.exists() or file.suffix != ".py":
            return f"Not a Python file: {path}"
        try:
            # -I: isolated mode, so modules next to the file can't shadow the standard library.
            result = subprocess.run(
                [sys.executable, "-I", str(file)],
                cwd=file.parent,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return f"The program did not finish within {TIMEOUT_SECONDS} seconds."
        output = f"Exit code: {result.returncode}\n--- stdout ---\n{result.stdout[-10_000:]}"
        if result.stderr:
            output += f"\n--- stderr ---\n{result.stderr[-5_000:]}"
        return output
