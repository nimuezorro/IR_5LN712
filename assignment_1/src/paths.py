"""Centralized filesystem paths for the CHILDES L1 project.

All paths are derived from this file's location instead of the current working
directory. That keeps scripts, inference, and the Gradio app usable whether they
are launched from the project root, from an IDE, or from another directory.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HF_DATASET_DIR = DATA_DIR / "hf_dataset"

RAW_CHILDES_DIR = RAW_DATA_DIR / "childes_archive"
GUIDE_CSV_PATH = RAW_CHILDES_DIR / "guide_to_files.csv"
PROCESSED_DATASET_CSV = PROCESSED_DATA_DIR / "childes_l1_dataset.csv"
PROCESSED_DATASET_JSONL = PROCESSED_DATA_DIR / "childes_l1_dataset.jsonl"
HF_DATASET_PATH = HF_DATASET_DIR / "childes_l1_dataset"
BALANCED_DATASET_CSV = PROCESSED_DATA_DIR / "childes_l1_dataset_balanced.csv"
BALANCED_DATASET_JSONL = PROCESSED_DATA_DIR / "childes_l1_dataset_balanced.jsonl"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
APP_DIR = PROJECT_ROOT / "app"

RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
LOGISTIC_REGRESSION_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
