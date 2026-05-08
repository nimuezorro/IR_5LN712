"""Hugging Face Space entry point for the CHILDES L1 classifier."""

from __future__ import annotations

import gradio as gr

try:
    from .inference import predict
except ImportError:
    # When Hugging Face Spaces or `python app/app.py` run this file directly,
    # the app directory is placed on sys.path and a local import is appropriate.
    from inference import predict


def predict_l1(text: str):
    """Format reusable inference output for Gradio components."""

    result = predict(text)
    if result["error"]:
        return result["error"], {}
    return result["prediction"], result["probabilities"]


with gr.Blocks(title="Child L1 Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# L1 Prediction from Bilingual Child Speech")

    with gr.Row():
        text_input = gr.Textbox(
            label="Child English speech sample",
            placeholder="Type or paste a child English utterance...",
            lines=6,
        )

    predict_button = gr.Button("Predict", variant="primary")

    with gr.Row():
        predicted_output = gr.Textbox(label="Predicted first language")
        confidence_output = gr.Label(label="Confidence scores", num_top_classes=9)

    predict_button.click(
        fn=predict_l1,
        inputs=text_input,
        outputs=[predicted_output, confidence_output],
    )
    text_input.submit(
        fn=predict_l1,
        inputs=text_input,
        outputs=[predicted_output, confidence_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
