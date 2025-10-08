# Demonstration of Strands Agents tool calling capabilities with Ollama and IBM Granite 4.0 SML.
# This example uses the calculator and current_time tools to enhance the model's capabilities.
# Ensure you have the Ollama server running locally with the IBM Granite 4.0 SML model downloaded.

import gradio as gr
from strands import Agent
from strands.models.ollama import OllamaModel
from strands_tools import calculator, current_time

OLLAMA_HOST = "http://localhost:11434"
MODEL_ID = "ibm/granite4:small-h"
TEMPERATURE = 0.2

DEFAULT_PROMPT = """Evaluate ((313*(451+293))/(4^2))+(sqrt(734)).
Subtract the results, rounded to the nearest integer, from the product of the current year and month.
Summarize your mathematical expressions work using LaTeX. Wrap each block of LaTeX in double dollar signs ('$$')."""

ollama_model = OllamaModel(
    host=OLLAMA_HOST,
    model_id=MODEL_ID,
    temperature=TEMPERATURE,
)

agent = Agent(
    model=ollama_model,
    tools=[current_time, calculator],
)


def prompt_agent(prompt):
    response = agent(prompt=prompt)
    return f"{response.message["content"][0]["text"]}"


with gr.Blocks(css="footer{display:none !important}") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Strands Agents | Ollama | Granite 4.0")
            input = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT)
            submit_btn = gr.Button("Submit", variant="primary")
    with gr.Row():
        with gr.Column(scale=1):
            output = gr.Markdown(
                value="Response will be appear here...",
                min_height=200,
                line_breaks=False,
            )
            submit_btn.click(fn=prompt_agent, inputs=[input], outputs=[output])

demo.launch()
