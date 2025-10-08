#!/usr/bin/env python3
"""
Strands Agent using Ollama and IBM Granite model for math calculations and LaTeX formatting.

This script sets up a Gradio interface to interact with a Strands Agent powered by
an Ollama-served IBM Granite model. The agent can perform mathematical calculations
and format responses using LaTeX.
"""

import logging

import gradio as gr
from strands import Agent
from strands.models.ollama import OllamaModel
from strands_tools import calculator, current_time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_HOST = "http://localhost:11434"
MODEL_ID = "ibm/granite4:small-h"
TEMPERATURE = 0.5

# Default sample prompt
DEFAULT_PROMPT = """Evaluate ((313*(451+293))/(4^2))+(sqrt(734)). Subtract the results, rounded to the nearest integer, from the product of the current year and month.
Summarize your mathematical expressions work using LaTeX. Wrap each section of LaTeX in double dollar signs ('$$') for Gradio."""


def create_model() -> OllamaModel:
    """
    Create and configure the Ollama model.

    Returns:
        OllamaModel: Configured model instance
    """
    return OllamaModel(
        host=OLLAMA_HOST,
        model_id=MODEL_ID,
        temperature=TEMPERATURE,
    )


def create_agent(model: OllamaModel) -> Agent:
    """
    Create a Strands Agent with the provided model and tools.

    Args:
        model: The language model to use for the agent

    Returns:
        Agent: Configured agent instance
    """
    return Agent(
        model=model,
        tools=[current_time, calculator],
    )


def process_prompt(prompt: str, agent: Agent) -> str:
    """
    Process a user prompt using the Strands Agent.

    Args:
        prompt: The user input prompt
        agent: The Strands Agent to use for processing

    Returns:
        str: The agent's response

    Raises:
        Exception: If there's an error processing the prompt
    """
    try:
        response = agent(prompt=prompt)
        if not response or not response.message:
            return "Error: No response received from agent"

        # Extract text content from the message
        try:
            # First try the direct approach used in other scripts
            return response.message["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return "Error: Invalid response format"
    except Exception as e:
        logger.error(f"Error processing prompt: {e}", exc_info=True)
        return f"Error: {str(e)}"


def build_gradio_interface(agent: Agent) -> gr.Blocks:
    """
    Build the Gradio interface for the Strands Agent.

    Args:
        agent: The configured Strands Agent

    Returns:
        gr.Blocks: The Gradio interface
    """
    with gr.Blocks(theme=gr.themes.Default()) as demo:
        gr.Markdown("# Strands Agent with IBM Granite 4.0")
        gr.Markdown("Interact with a Strands Agent running IBM Granite 4.0 via Ollama")

        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Input Prompt",
                    placeholder="Enter your prompt here...",
                    value=DEFAULT_PROMPT,
                    lines=5,
                )

                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Row():
            output_text = gr.Markdown(
                value="Response will appear here...",
                label="Response",
                elem_id="output",
                min_height=300,
            )

        # Define callbacks
        submit_btn.click(
            fn=lambda prompt: process_prompt(prompt, agent),
            inputs=[input_text],
            outputs=[output_text],
            show_progress=True,
        )

        clear_btn.click(
            fn=lambda: ("", "Response will appear here..."),
            outputs=[input_text, output_text],
        )

        # Add examples
        gr.Examples(
            examples=[
                [DEFAULT_PROMPT],
                [
                    "What is the square root of 144? Summarize your mathematical expressions work using LaTeX. Wrap each section of LaTeX in double dollar signs ('$$')."
                ],
                [
                    "Calculate the integral of x^2 from 0 to 3. Summarize your mathematical expressions work using LaTeX. Wrap each section of LaTeX in double dollar signs ('$$')."
                ],
            ],
            inputs=[input_text],
        )

    return demo


def main() -> None:
    """
    Main entry point for the application.
    """
    try:
        # Create model, agent, and interface
        model = create_model()
        agent = create_agent(model)
        demo = build_gradio_interface(agent)

        # Launch the Gradio interface
        demo.launch()
    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
