from pathlib import Path

from smolagents import Tool

from tools.llm import transcribe


class TranscribeAudioTool(Tool):
    name = "transcribe_audio"
    description = "Transcribes a local audio file (mp3, wav, m4a, ...) to text with a speech-to-text model."
    inputs = {"path": {"type": "string", "description": "Path of the local audio file."}}
    output_type = "string"

    def forward(self, path: str) -> str:
        file = Path(path)
        if not file.exists():
            return f"File not found: {path}"
        # Transcripts are cached next to the audio so repeated runs cost nothing.
        cached = file.with_suffix(file.suffix + ".txt")
        if cached.exists():
            return cached.read_text()
        try:
            text = transcribe(str(file))
        except Exception as e:
            return f"Transcription failed: {e}"
        cached.write_text(text)
        return text
