"""Gradio app for predicting a child's likely L1 from English text.

This file is designed for Hugging Face Spaces.  The app loads the same
Sentence-BERT embedding model used during training, loads the saved scikit-learn
classifier, and returns both the predicted first language and class confidence
scores.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CLASSIFIER_PATH = MODEL_DIR / "random_forest.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"


@lru_cache(maxsize=1)
def load_resources():
    """Load model resources once and reuse them for all predictions."""

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    classifier = joblib.load(CLASSIFIER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return embedding_model, classifier, label_encoder


def softmax(values: np.ndarray) -> np.ndarray:
    """Convert arbitrary model scores to probabilities when needed."""

    values = values.astype(float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def confidence_scores(classifier, embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return encoded classes and confidence scores for one prediction."""

    if hasattr(classifier, "predict_proba"):
        return classifier.classes_, classifier.predict_proba(embedding)[0]

    if hasattr(classifier, "decision_function"):
        scores = classifier.decision_function(embedding)
        scores = np.atleast_2d(scores)[0]
        return classifier.classes_, softmax(scores)

    prediction = classifier.predict(embedding)[0]
    scores = np.zeros(len(classifier.classes_), dtype=float)
    prediction_index = np.where(classifier.classes_ == prediction)[0][0]
    scores[prediction_index] = 1.0
    return classifier.classes_, scores


def predict_l1(text: str):
    """Predict the likely first language for one English text sample."""

    if not text or not text.strip():
        return "Please enter some English text.", {}

    embedding_model, classifier, label_encoder = load_resources()
    embedding = embedding_model.encode(
        [text.strip()],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    encoded_classes, probabilities = confidence_scores(classifier, embedding)
    labels = label_encoder.inverse_transform(encoded_classes.astype(int))
    scores = {
        label: float(probability)
        for label, probability in sorted(
            zip(labels, probabilities, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    predicted_class = next(iter(scores))
    return predicted_class, scores


with gr.Blocks(title="Child L1 Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Child L1 Classifier")

    with gr.Row():
        text_input = gr.Textbox(
            label="English text",
            placeholder="Type or paste a child English utterance here...",
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
    demo.launch()
