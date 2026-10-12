"""Direct calls to the provider for tools that need vision or audio models."""
import base64
import mimetypes
import os

from openai import OpenAI


def client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("OPENAI_API_BASE"), api_key=os.getenv("OPENAI_API_KEY"), timeout=120, max_retries=2
    )


def ask_vision(prompt: str, image: bytes, mime_type: str = "image/jpeg") -> str:
    data_url = f"data:{mime_type};base64," + base64.b64encode(image).decode()
    response = client().chat.completions.create(
        model=os.getenv("VISION_MODEL_ID") or os.getenv("MODEL_ID", "gpt-5-mini"),
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
        ]}],
    )
    return response.choices[0].message.content.strip()


def ask_vision_file(prompt: str, path: str) -> str:
    with open(path, "rb") as f:
        return ask_vision(prompt, f.read(), mimetypes.guess_type(path)[0] or "image/png")


def transcribe(path: str) -> str:
    with open(path, "rb") as f:
        result = client().audio.transcriptions.create(
            model=os.getenv("TRANSCRIBE_MODEL_ID", "whisper-1"), file=f
        )
    return result.text.strip()
