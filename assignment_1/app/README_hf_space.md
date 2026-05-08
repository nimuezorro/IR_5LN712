---
title: Bilingual Children Speech L1 Classifier
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
python_version: 3.12
app_file: app/app.py
pinned: false
---

# Bilingual Children Speech L1 Classifier

This Space demonstrates a text classification model for predicting a bilingual
child's first language (L1) from English speech samples. The project is intended
as an academic machine-learning pipeline built from CHILDES/CHAT transcript data.

## Project Goal

The goal is to test whether short samples of bilingual child English speech
contain distributional signals that can help predict the child's L1. The task is
framed as supervised multiclass classification: the input is a cleaned English
utterance chunk, and the output is an L1 label.

## Method

Input text is embedded with
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. This multilingual
Sentence-BERT model maps each text sample to a fixed-length dense vector, making
it possible to use a conventional scikit-learn classifier on variable-length
child speech.

The deployed classifier is a trained Random Forest. Random Forest was used
because it is CPU-compatible, handles nonlinear feature interactions, requires
little preprocessing after embedding, and can provide class probabilities through
`predict_proba`. The model architecture and trained weights are not modified in
this Space.

## Dataset

The dataset was created from bilingual children's speech transcripts in CHILDES
CHAT format. Only target-child (`CHI`) utterances were used. Utterances were
cleaned, short uninformative fragments were removed, and consecutive child
utterances were grouped into medium-length text chunks. Each example contains a
text sample and an L1 label, with speaker metadata used during training to avoid
placing the same child in both train and test splits.

## Files

- `app.py`: Gradio Space entry point.
- `inference.py`: reusable prediction wrapper with `predict(text: str) -> dict`.
- `models/random_forest.joblib`: trained Random Forest classifier.
- `models/label_encoder.joblib`: fitted label encoder for L1 labels.
- `requirements.txt`: lightweight CPU-compatible dependencies.

## Run Locally

Install the Space dependencies:

```bash
pip install -r requirements.txt
```

Run the app from this directory:

```bash
gradio app.py
```

You can also call inference directly:

```python
from inference import predict

result = predict("I want to play with the toys and then go outside")
print(result)
```

## Hugging Face Spaces

This folder is ready to upload as a Gradio Space. The app uses only relative
paths, loads the trained model from `models/`, downloads the Sentence-BERT
embedding model at startup if needed, and runs on CPU.
