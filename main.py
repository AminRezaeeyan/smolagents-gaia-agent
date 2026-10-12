"""Entry point: solves questions with GaiaAgent and submits answers.

Examples:
    python main.py --list                # show the questions with their numbers
    python main.py --only 1              # solve question 1 only
    python main.py --only 1,5 --force    # re-solve 1 and 5 even if already answered
    python main.py                       # solve every question that has no cached answer yet
    python main.py --submit              # submit all cached answers (no solving)
"""
import argparse
import os

from dotenv import load_dotenv

import gaia_api

load_dotenv()


def select_questions(questions: list[dict], only: str | None) -> list[tuple[int, dict]]:
    """`only` is a comma separated list of question numbers (1-based) and/or task IDs."""
    numbered = list(enumerate(questions, start=1))
    if not only:
        return numbered
    wanted = {w.strip() for w in only.split(",") if w.strip()}
    return [(i, q) for i, q in numbered if str(i) in wanted or q["task_id"] in wanted]


def solve(only: str | None = None, force: bool = False) -> list[dict]:
    """Runs the agent on the selected questions, caching each answer as soon as it is produced."""
    from agent import GaiaAgent

    questions = select_questions(gaia_api.get_questions(), only)
    cached = gaia_api.load_answers()
    agent = None
    rows = []
    for number, q in questions:
        task_id = q["task_id"]
        if task_id in cached and not force:
            print(f"[{number}] cached -> {cached[task_id]!r}")
            rows.append({"#": number, "Task ID": task_id, "Question": q["question"], "Answer": cached[task_id]})
            continue
        agent = agent or GaiaAgent()
        print(f"\n[{number}] {q['question']}")
        try:
            answer = agent(q["question"], gaia_api.get_file(q))
            gaia_api.save_answer(task_id, answer)
        except Exception as e:
            answer = f"AGENT ERROR: {e}"
        print(f"[{number}] answer -> {answer!r}")
        rows.append({"#": number, "Task ID": task_id, "Question": q["question"], "Answer": answer})
    return rows


def submit(username: str, agent_code: str) -> str:
    answers = gaia_api.load_answers()
    if not answers:
        return "No cached answers to submit. Solve some questions first."
    result = gaia_api.submit(username, agent_code, answers)
    return (
        f"Submitted {len(answers)} answers for '{result.get('username')}'\n"
        f"Score: {result.get('score', 'N/A')}% "
        f"({result.get('correct_count', '?')}/{result.get('total_attempted', '?')} correct)\n"
        f"{result.get('message', '')}"
    )


def run_and_submit(username: str, agent_code: str, only: str | None = None, force: bool = False):
    rows = solve(only, force)
    return submit(username, agent_code), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="Question numbers and/or task IDs, comma separated (e.g. 1,3).")
    parser.add_argument("--force", action="store_true", help="Re-solve even if an answer is cached.")
    parser.add_argument("--submit", action="store_true", help="Only submit cached answers.")
    parser.add_argument("--list", action="store_true", help="List questions and cached answers.")
    parser.add_argument("--refresh", action="store_true", help="Re-download the questions.")
    args = parser.parse_args()

    questions = gaia_api.get_questions(refresh=args.refresh)

    if args.list:
        cached = gaia_api.load_answers()
        for number, q in select_questions(questions, args.only):
            file_note = f" [file: {q['file_name']}]" if q.get("file_name") else ""
            print(f"{number:>2}. {q['question'][:110]!r}{file_note}")
            print(f"    answer: {cached.get(q['task_id'], '-')}")
        return

    if args.submit:
        username, agent_code = os.getenv("HF_USERNAME"), os.getenv("AGENT_CODE_URL")
        if not username or not agent_code:
            raise SystemExit("Set HF_USERNAME and AGENT_CODE_URL in .env before submitting.")
        print(submit(username, agent_code))
        return

    solve(args.only, args.force)


if __name__ == "__main__":
    main()
