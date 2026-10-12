import os
import re

from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel

from tools import get_tools

load_dotenv()

INSTRUCTIONS = """You are answering questions from the GAIA benchmark. Answers are graded by EXACT MATCH.
When you call final_answer, pass ONLY the answer itself:
- A number: digits only, no commas, no units (unless the question asks for them).
- A string: as few words as possible, no articles, no abbreviations unless asked.
- A list: comma separated, following the rules above for each element.
- Never include explanations, "FINAL ANSWER:", or surrounding punctuation.
Follow any formatting the question requests exactly.
Verify facts with your tools instead of relying on memory. Search snippets are only leads:
open the source page (visit_webpage / wikipedia_page) and read it before answering.
If the question is about a specific point in time ("as of July 2023", "in 2022"), the current
version of a page may be different: use as_of_date to read the page as it was at that time.
For classification questions, go through EVERY item one by one in your reasoning (item -> category
-> why), applying the exact definition the question asks for (e.g. a scientific one) rather than
everyday usage, and only then build the answer. If the question refers to a specific
version of a source (e.g. "2022 version of Wikipedia"), use that version.
Be careful when extracting data with code: pages contain several similar tables/sections
(e.g. studio vs live vs compilation albums). Print and read the exact rows you count, check they
belong to the right category, and only call final_answer in a later step after reviewing them.
Do not merge distinct categories when aggregating (e.g. two species of the same family are two
species). For visual questions, re-examine the decisive moments with a closer, denser look
before answering."""


class GaiaAgent:
    def __init__(self, max_steps: int = 12):
        model = OpenAIServerModel(
            model_id=os.getenv("MODEL_ID", "gpt-5-mini"),
            api_base=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.agent = CodeAgent(
            tools=get_tools(),
            model=model,
            instructions=INSTRUCTIONS,
            max_steps=max_steps,
            executor_kwargs={"timeout_seconds": 600},  # video/vision tools can take minutes
            additional_authorized_imports=["re", "json", "math", "datetime", "collections", "itertools", "pandas", "numpy"],
        )

    def __call__(self, question: str, file_path: str | None = None) -> str:
        task = question
        if file_path:
            task += (
                f"\n\nThe file attached to this question is available at: {file_path}\n"
                "Use the tool matching its type (read_file, run_python_file, transcribe_audio, "
                "analyze_image, chess_board_from_image)."
            )
        answer = self.agent.run(task)
        return clean_answer(str(answer))


def clean_answer(answer: str) -> str:
    answer = re.sub(r"^\s*final answer\s*:\s*", "", answer, flags=re.IGNORECASE)
    return answer.strip().strip('"').strip()
