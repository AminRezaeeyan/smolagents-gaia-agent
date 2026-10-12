"""Talks to the course scoring server and caches what it returns under cache/."""
import json
import shutil
from pathlib import Path

import requests

API_URL = "https://agents-course-unit4-scoring.hf.space"
CACHE_DIR = Path(__file__).parent / "cache"
QUESTIONS_FILE = CACHE_DIR / "questions.json"
ANSWERS_FILE = CACHE_DIR / "answers.json"
FILES_DIR = CACHE_DIR / "files"
GAIA_DATASET = "gaia-benchmark/GAIA"


def get_questions(refresh: bool = False) -> list[dict]:
    if QUESTIONS_FILE.exists() and not refresh:
        return json.loads(QUESTIONS_FILE.read_text())
    response = requests.get(f"{API_URL}/questions", timeout=30)
    response.raise_for_status()
    questions = response.json()
    CACHE_DIR.mkdir(exist_ok=True)
    QUESTIONS_FILE.write_text(json.dumps(questions, indent=2, ensure_ascii=False))
    return questions


def get_file(question: dict) -> str | None:
    """Downloads the question's attachment (if any) and returns its local path."""
    file_name = question.get("file_name")
    if not file_name:
        return None
    path = FILES_DIR / file_name
    if not path.exists():
        FILES_DIR.mkdir(parents=True, exist_ok=True)
        response = requests.get(f"{API_URL}/files/{question['task_id']}", timeout=60)
        if response.ok:
            path.write_bytes(response.content)
        else:
            # The course server is missing some files; the same ones are in the (gated) GAIA dataset.
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(GAIA_DATASET, f"2023/validation/{file_name}", repo_type="dataset")
            shutil.copy(downloaded, path)
    return str(path)


def load_answers() -> dict[str, str]:
    if ANSWERS_FILE.exists():
        return json.loads(ANSWERS_FILE.read_text())
    return {}


def save_answer(task_id: str, answer: str) -> None:
    answers = load_answers()
    answers[task_id] = answer
    CACHE_DIR.mkdir(exist_ok=True)
    ANSWERS_FILE.write_text(json.dumps(answers, indent=2, ensure_ascii=False))


def submit(username: str, agent_code: str, answers: dict[str, str]) -> dict:
    payload = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": [{"task_id": t, "submitted_answer": a} for t, a in answers.items()],
    }
    response = requests.post(f"{API_URL}/submit", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()
