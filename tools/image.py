from pathlib import Path

from smolagents import Tool

from tools.llm import ask_vision_file


class AnalyzeImageTool(Tool):
    name = "analyze_image"
    description = (
        "Answers a question about a local image file using a vision model "
        "(describe it, read text, count objects, ...). For chess positions use chess_board_from_image."
    )
    inputs = {
        "path": {"type": "string", "description": "Path of the local image file."},
        "question": {"type": "string", "description": "What you want to know about the image."},
    }
    output_type = "string"

    def forward(self, path: str, question: str) -> str:
        if not Path(path).exists():
            return f"File not found: {path}"
        try:
            return ask_vision_file(question, path)
        except Exception as e:
            return f"Image analysis failed: {e}"
