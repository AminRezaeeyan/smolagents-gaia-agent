import os

import gradio as gr
import pandas as pd

from main import run_and_submit


def on_run(profile: gr.OAuthProfile | None):
    if profile is None:
        return "Please log in to Hugging Face with the button.", None
    agent_code = f"https://huggingface.co/spaces/{os.getenv('SPACE_ID')}/tree/main"
    try:
        status, rows = run_and_submit(profile.username, agent_code)
    except Exception as e:
        return f"Failed: {e}", None
    return status, pd.DataFrame(rows)


with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
    run_button.click(fn=on_run, outputs=[status_output, results_table])

if __name__ == "__main__":
    demo.launch(debug=True, share=False)
