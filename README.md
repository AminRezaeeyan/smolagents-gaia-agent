# smolagents GAIA Agent

A tool-using LLM agent built with [smolagents](https://github.com/huggingface/smolagents) for the final assignment of the [Hugging Face Agents Course](https://huggingface.co/learn/agents-course). It answers 20 questions from the [GAIA benchmark](https://huggingface.co/papers/2311.12983), which require web research, reading Wikipedia as it was at a given date, watching YouTube videos, listening to audio, reading spreadsheets, running code and solving a chess position.

**Result: 100% (20/20 correct)** on the course leaderboard.

🤗 Live Space: [AminR2005/Final_Assignment_Template](https://huggingface.co/spaces/AminR2005/Final_Assignment_Template)

## How it works

```
question ──► CodeAgent (gpt-5-mini) ──► writes Python that calls tools ──► final_answer
                    │
                    ├─ web_search ─────────── DuckDuckGo results
                    ├─ visit_webpage ──────── any page, or its Wayback Machine snapshot at a date
                    ├─ wikipedia_page ─────── full article, optionally as of a date / one section
                    ├─ youtube_transcript ─── captions with timestamps
                    ├─ youtube_video_analysis  frames sampled and described by a vision model
                    ├─ read_file ──────────── text, code, CSV, Excel
                    ├─ run_python_file ────── runs an attached script in a separate process
                    ├─ transcribe_audio ───── Whisper speech-to-text
                    ├─ analyze_image ──────── vision model Q&A on an image
                    ├─ chess_board_from_image  board image ──► FEN
                    └─ chess_best_move ────── Stockfish analysis
```

The agent is a smolagents `CodeAgent`: instead of emitting JSON tool calls, it writes short Python snippets that call the tools and process their output (parse tables, count, sort), then calls `final_answer`. GAIA is graded by exact match, so the system prompt in [agent.py](agent.py) enforces a strict answer format and pushes the model to verify facts from the source rather than from search snippets or memory.

Some design choices that mattered for accuracy:

- **Time-pinned sources.** Questions like "as of July 2023" or "the 2022 version of Wikipedia" are answered from the page *as it was then*: `wikipedia_page` fetches the revision on a date, `visit_webpage` fetches the Wayback Machine snapshot. Search results carry a reminder that snippets show today's version.
- **Video analysis in batches.** Frames are sampled every few seconds, tiled 4 per image with timestamps and sent to the vision model in parallel. The agent can re-check a time range at a higher sampling rate, and pass what it already learned (`context`) so a species identified in a close-up is recognised in wide shots too.
- **Reliable chess board reading.** Each square of the board image is labelled with a number (1–64) before it is sent to the vision model, so the model reports pieces per labelled square instead of counting squares; code then maps the labels to board coordinates using the detected orientation. The position is validated with `python-chess` and analysed by Stockfish.
- **Caching everywhere.** Questions, attachments, videos, transcripts and answers are cached under `cache/`, so re-running a question you already solved costs no tokens.

## Project structure

```
├── app.py              # Gradio UI for the Hugging Face Space (login + run button only)
├── main.py             # Entry point and CLI: select questions, solve, cache, submit
├── agent.py            # GaiaAgent: CodeAgent, model config and system instructions
├── gaia_api.py         # Scoring server client: questions, attachments, answers, submit
├── tools/
│   ├── __init__.py     # Tool registry
│   ├── llm.py          # Shared vision and transcription calls
│   ├── web_search.py
│   ├── visit_webpage.py
│   ├── wikipedia.py
│   ├── youtube.py
│   ├── files.py
│   ├── python_runner.py
│   ├── audio.py
│   ├── image.py
│   └── chess_tools.py
├── requirements.txt    # Python dependencies
└── packages.txt        # System packages for the Space (stockfish)
```

## Setup

Requirements: Python 3.10+, and Node.js (used by `yt-dlp` to read YouTube).

```bash
git clone https://github.com/AminRezaeeyan/smolagents-gaia-agent.git
cd smolagents-gaia-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in your values
```

### Configuration (`.env`)

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_BASE` | yes | Base URL of any OpenAI-compatible API |
| `OPENAI_API_KEY` | yes | API key for that provider |
| `MODEL_ID` | yes | Agent model, e.g. `gpt-5-mini` (must support images for the vision tools) |
| `HF_USERNAME` | for submitting | Your Hugging Face username |
| `AGENT_CODE_URL` | for submitting | Public link to your code |
| `VISION_MODEL_ID` | no | Separate model for image/video tools (defaults to `MODEL_ID`) |
| `TRANSCRIBE_MODEL_ID` | no | Speech-to-text model (default `whisper-1`) |
| `PROXY_URL` | no | Proxy for YouTube downloads and transcripts, e.g. `socks5://127.0.0.1:10808` |
| `STOCKFISH_PATH` | no | Path to the Stockfish binary if it is not on your `PATH` |

### Stockfish

The chess tool needs the [Stockfish](https://stockfishchess.org/download/) engine. Install it with your package manager (`sudo apt install stockfish`, `brew install stockfish`), or download a release binary and either put it at `cache/bin/stockfish` or set `STOCKFISH_PATH`.

### Question attachments

Some questions come with a file (image, audio, Excel, Python). If the course server does not provide it, the files are downloaded from the official [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA), which is gated: accept its terms on Hugging Face and log in with `hf auth login`.

## Usage

```bash
python main.py --list               # list the questions and their cached answers
python main.py --only 1             # solve question 1 only
python main.py --only 2,7           # solve several (numbers or task IDs)
python main.py --only 4 --force     # re-solve a question that already has an answer
python main.py                      # solve every question without a cached answer
python main.py --submit             # submit all cached answers and print the score
python main.py --refresh --list     # re-download the question list
```

Answers are saved to `cache/answers.json` as soon as each question is solved, so you can work through the questions one at a time and submit whenever you like.

### Hugging Face Space

`app.py` is a thin Gradio interface that logs in with Hugging Face and calls the same `run_and_submit` used by the CLI. To deploy it, push the code to a Gradio Space and add `OPENAI_API_KEY` as a secret and `OPENAI_API_BASE` / `MODEL_ID` as variables in the Space settings. YouTube may block video downloads from cloud servers, so running locally is more reliable.

## Notes

- `cache/` is git-ignored on purpose: it contains GAIA questions and answers, which the benchmark authors ask not to publish, to keep them out of training data.
- `run_python_file` executes the attached script on your machine (in a separate process, with a timeout). Review untrusted files before running them.
- Results from vision-based tools can vary between runs; if an answer looks wrong, re-run that question with `--force`.
