"""Reusable inference utilities for the CHILDES L1 classifier Space."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


# Resolve paths from this file so inference works from any working directory.
# The fallback keeps the app usable if the app folder is copied into a Space
# together with a sibling `models/` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.paths import LABEL_ENCODER_PATH, RANDOM_FOREST_MODEL_PATH
except ModuleNotFoundError:
    fallback_models_dir = PROJECT_ROOT / "models"
    RANDOM_FOREST_MODEL_PATH = fallback_models_dir / "random_forest.joblib"
    LABEL_ENCODER_PATH = fallback_models_dir / "label_encoder.joblib"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CLASSIFIER_PATH = RANDOM_FOREST_MODEL_PATH


@lru_cache(maxsize=1)
def load_resources() -> tuple[SentenceTransformer, Any, Any]:
    """Load the embedding model, Random Forest classifier, and label encoder."""

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Classifier file not found: {CLASSIFIER_PATH}")
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder file not found: {LABEL_ENCODER_PATH}")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    classifier = joblib.load(CLASSIFIER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return embedding_model, classifier, label_encoder


def _softmax(values: np.ndarray) -> np.ndarray:
    """Convert decision scores to probability-like confidence values."""

    values = values.astype(float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def _confidence_scores(classifier: Any, embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return encoded class IDs and one confidence value per class."""

    if hasattr(classifier, "predict_proba"):
        return classifier.classes_, classifier.predict_proba(embedding)[0]

    if hasattr(classifier, "decision_function"):
        scores = np.atleast_2d(classifier.decision_function(embedding))[0]
        return classifier.classes_, _softmax(scores)

    prediction = classifier.predict(embedding)[0]
    scores = np.zeros(len(classifier.classes_), dtype=float)
    prediction_index = np.where(classifier.classes_ == prediction)[0][0]
    scores[prediction_index] = 1.0
    return classifier.classes_, scores


def predict(text: str) -> dict[str, Any]:
    """Predict a child's likely L1 from an English speech sample."""

    clean_text = text.strip() if text else ""
    if not clean_text:
        return {
            "prediction": None,
            "probabilities": {},
            "error": "Please enter some English text.",
        }

    embedding_model, classifier, label_encoder = load_resources()
    embedding = embedding_model.encode(
        [clean_text],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    encoded_classes, probabilities = _confidence_scores(classifier, embedding)
    labels = label_encoder.inverse_transform(encoded_classes.astype(int))
    score_items = sorted(
        zip(labels, probabilities, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    scores = {label: float(probability) for label, probability in score_items}

    return {
        "prediction": score_items[0][0],
        "probabilities": scores,
        "error": None,
    }
