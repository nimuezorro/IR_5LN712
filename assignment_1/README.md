# CHILDES L1 Classification Project

This project builds a supervised text classification dataset from CHILDES/CHAT
bilingual child speech transcripts, trains scikit-learn classifiers, and serves
the trained Random Forest model in a Gradio demo.

## Assignment Links
- [HF Classifier](https://huggingface.co/nimuezorro/bilingual_children_speech_classifier)
- [HF Random Forest Demo](https://huggingface.co/spaces/nimuezorro/bilingual_children_speech_random_forest_demo)
- [HF Dataset](https://huggingface.co/datasets/nimuezorro/bilingual_children_speech)

## Project Structure

```text
assignment_1/
├── app/                  # Gradio demo and inference wrapper
├── data/
│   ├── raw/              # Original CHILDES .cha files and guide CSV
│   ├── processed/        # Cleaned CSV and JSONL dataset files
│   └── hf_dataset/       # Hugging Face datasets save_to_disk output
├── models/               # Trained sklearn models and label encoder
├── reports/              # Metrics, comparisons, and figures
├── scripts/              # Data-building and training scripts
├── src/                  # Shared project code and centralized paths
├── requirements.txt
└── .gitignore
```

All path constants live in `src/paths.py`. They are derived with
`Path(__file__).resolve()`, so scripts and the app do not depend on the current
working directory.

## Setup

```bash
cd /IR_5LN712/assignment_1
pip install -r requirements.txt
```

## Build Dataset 

This is how I extracted the CHILDES transcripts into a functional dataset that includes only child utterances. The utterances themself were combined and turned into medium-length text chunks. Since the original data is a dialogue and contains lots of short utterances. 

```bash
python scripts/build_childes_l1_dataset.py
```

Defaults:

- raw CHAT files: `data/raw/childes_archive`
- guide CSV: `data/raw/childes_archive/guide_to_files.csv`
- processed CSV/JSONL: `data/processed`
- Hugging Face dataset: `data/hf_dataset/childes_l1_dataset`

The old wrapper still works:

```bash
python build_childes_l1_dataset.py
```

`data/processed/childes_l1_dataset_balanced.*` contains synthetically created data to balance the existing dataset

## Train Models

Models use `data/processed/childes_l1_dataset_balanced.*` dataset by default.

```bash
python scripts/train_childes_l1_models.py
```

Defaults:

- input CSV: `data/processed/childes_l1_dataset_balanced.csv`
- trained models: `models/`
- metrics and comparison reports: `reports/`
- confusion matrices: `reports/figures/`

The old wrapper still works:

```bash
python train_childes_l1_models.py
```

## Run Inference

```python
from app.inference import predict

result = predict("I want to play with the toys and then go outside")
print(result)
```

## Run Gradio App

```bash
python app/app.py
```

For Hugging Face Spaces, upload the project root and use `app/app.py` as the
Space entry point. The app loads `models/random_forest.joblib` and
`models/label_encoder.joblib` through the centralized path config.
