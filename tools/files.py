from pathlib import Path

from smolagents import Tool

MAX_CHARS = 40_000


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Reads a local file attached to the question and returns its content as text. "
        "Supports text/code files (.py, .txt, .md, .json, ...), CSV and Excel (.xlsx/.xls: every "
        "sheet as a table). For images use analyze_image, for audio use transcribe_audio."
    )
    inputs = {"path": {"type": "string", "description": "Path of the local file."}}
    output_type = "string"

    def forward(self, path: str) -> str:
        file = Path(path)
        if not file.exists():
            return f"File not found: {path}"
        suffix = file.suffix.lower()
        try:
            if suffix in (".xlsx", ".xls", ".xlsm"):
                content = self._read_excel(file)
            elif suffix == ".csv":
                import pandas as pd

                content = pd.read_csv(file).to_markdown(index=False)
            else:
                content = file.read_text(errors="replace")
        except Exception as e:
            return f"Could not read {path}: {e}"
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + "\n\n...[truncated]"
        return content

    @staticmethod
    def _read_excel(file: Path) -> str:
        import pandas as pd

        sheets = pd.read_excel(file, sheet_name=None)
        return "\n\n".join(
            f"## Sheet: {name} ({len(df)} rows)\n\n{df.to_markdown(index=False)}" for name, df in sheets.items()
        )
