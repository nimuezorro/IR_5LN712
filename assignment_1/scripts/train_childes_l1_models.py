#!/usr/bin/env python3
"""Train L1 classifiers on the cleaned CHILDES dataset.

The preprocessing script creates one row per child-speech text chunk.  This
training script turns each text chunk into a multilingual Sentence-BERT
embedding and compares three standard classifiers:

* Logistic Regression
* Linear Support Vector Machine
* Random Forest

The most important methodological detail is the train/test split.  Multiple
text chunks can come from the same child, so a normal random row split would let
the same speaker appear in both train and test.  That would overestimate
performance because the model could learn speaker-specific patterns.  Instead,
we split by `child_id`, which gives a stricter and more academically defensible
estimate of generalization to unseen children.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Add the project root to sys.path based on this file's location. This keeps the
# shared path config import independent of the shell's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Matplotlib writes a font cache on import. Keep that cache inside the project
# so training works from restricted shells and IDE environments.
MPLCONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

from src.paths import FIGURES_DIR, MODELS_DIR, BALANCED_DATASET_CSV, REPORTS_DIR


EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RANDOM_STATE = 42


@dataclass(frozen=True)
class SplitInfo:
    """Metadata explaining how the grouped train/test split was made."""

    train_rows: int
    test_rows: int
    train_children: int
    test_children: int
    train_labels: list[str]
    test_labels: list[str]
    singleton_child_labels_kept_in_train: list[str]


def parse_args() -> argparse.Namespace:
    """Read command-line arguments with assignment-friendly defaults."""

    parser = argparse.ArgumentParser(
        description="Train and evaluate L1 classifiers with Sentence-BERT embeddings."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=BALANCED_DATASET_CSV,
        help="Path to the balanced CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory where metrics and comparison reports are saved.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory where trained .joblib models are saved.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory where confusion matrix figures are saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Approximate fraction of non-singleton child IDs used for testing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used by SentenceTransformer.encode().",
    )
    return parser.parse_args()


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and validate the cleaned CSV.

    Required columns:
    - `text`: the cleaned child utterance chunk used as model input.
    - `l1`: the first-language label to predict.
    - `child_id`: the grouping variable used to prevent speaker leakage.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"text", "l1", "child_id"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {missing}")

    df = df.dropna(subset=["text", "l1", "child_id"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["l1"] = df["l1"].astype(str).str.strip()
    df["child_id"] = df["child_id"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["l1"] != "") & (df["child_id"] != "")]

    if df.empty:
        raise ValueError("Dataset has no usable rows after removing empty values.")

    return df.reset_index(drop=True)


def grouped_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    """Split rows so that no real child appears in both train and test.

    Synthetic rows are ALWAYS placed into the training set and never used
    for evaluation. This prevents evaluation leakage because synthetic
    samples are generated from the original training distribution.
    """

    # ------------------------------------------------------------------
    # Separate synthetic rows from genuine child data
    # ------------------------------------------------------------------
    synthetic_mask = df["child_id"].str.lower() == "synthetic"

    synthetic_df = df[synthetic_mask].copy()
    real_df = df[~synthetic_mask].copy()

    if real_df.empty:
        raise ValueError("Dataset contains no real child data.")

    # ------------------------------------------------------------------
    # Validate that each real child has only one L1 label
    # ------------------------------------------------------------------
    child_labels = real_df.groupby("child_id")["l1"].nunique()

    mixed_label_children = child_labels[child_labels > 1]
    if not mixed_label_children.empty:
        children = ", ".join(mixed_label_children.index.tolist())
        raise ValueError(
            "Each child_id should map to exactly one l1 label. "
            f"Found mixed labels for: {children}"
        )

    # ------------------------------------------------------------------
    # Detect labels represented by only one real child
    # ------------------------------------------------------------------
    label_to_child_count = real_df.groupby("l1")["child_id"].nunique()

    singleton_labels = (
        label_to_child_count[label_to_child_count == 1]
        .index
        .tolist()
    )

    singleton_child_ids = set(
        real_df.loc[
            real_df["l1"].isin(singleton_labels),
            "child_id",
        ].unique().tolist()
    )

    singleton_df = real_df[
        real_df["child_id"].isin(singleton_child_ids)
    ]

    splittable_df = real_df[
        ~real_df["child_id"].isin(singleton_child_ids)
    ]

    if splittable_df["child_id"].nunique() < 2:
        raise ValueError(
            "Need at least two non-singleton child IDs "
            "to create a grouped test set."
        )

    # ------------------------------------------------------------------
    # Grouped split on REAL children only
    # ------------------------------------------------------------------
    splitter = GroupShuffleSplit(
        n_splits=200,
        test_size=test_size,
        random_state=random_state,
    )

    selected_split: tuple[np.ndarray, np.ndarray] | None = None

    all_labels = set(real_df["l1"].unique().tolist())

    for train_idx, test_idx in splitter.split(
        splittable_df,
        groups=splittable_df["child_id"],
    ):

        candidate_train = pd.concat(
            [
                singleton_df,
                splittable_df.iloc[train_idx],
                synthetic_df,  # ALWAYS train only
            ],
            ignore_index=True,
        )

        candidate_test = splittable_df.iloc[test_idx]

        train_labels = set(candidate_train["l1"].unique().tolist())
        test_labels = set(candidate_test["l1"].unique().tolist())

        if (
            all_labels.issubset(train_labels)
            and test_labels.issubset(train_labels)
        ):
            selected_split = (train_idx, test_idx)
            break

    if selected_split is None:
        raise ValueError(
            "Could not create a grouped split where all "
            "test labels are present in training. "
            "Try a smaller --test-size."
        )

    train_idx, test_idx = selected_split

    # ------------------------------------------------------------------
    # Final train/test datasets
    # ------------------------------------------------------------------
    train_df = pd.concat(
        [
            singleton_df,
            splittable_df.iloc[train_idx],
            synthetic_df,  # synthetic always train
        ],
        ignore_index=True,
    )

    test_df = splittable_df.iloc[test_idx].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Final leakage safety check
    # ------------------------------------------------------------------
    overlapping_children = set(
        train_df.loc[
            train_df["child_id"] != "synthetic",
            "child_id",
        ]
    ).intersection(
        test_df["child_id"]
    )

    if overlapping_children:
        raise RuntimeError(
            f"Grouped split failed; overlap: {overlapping_children}"
        )

    split_info = SplitInfo(
        train_rows=len(train_df),
        test_rows=len(test_df),
        train_children=train_df[
            train_df["child_id"] != "synthetic"
        ]["child_id"].nunique(),
        test_children=test_df["child_id"].nunique(),
        train_labels=sorted(train_df["l1"].unique().tolist()),
        test_labels=sorted(test_df["l1"].unique().tolist()),
        singleton_child_labels_kept_in_train=sorted(singleton_labels),
    )

    return (
        train_df.reset_index(drop=True),
        test_df,
        split_info,
    )

def encode_texts(
    train_texts: list[str],
    test_texts: list[str],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode text chunks with a multilingual Sentence-BERT model."""

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    train_embeddings = model.encode(
        train_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    test_embeddings = model.encode(
        test_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return train_embeddings, test_embeddings


def build_classifiers() -> dict[str, Pipeline | RandomForestClassifier]:
    """Create comparable classifiers using scikit-learn best practices.

    Logistic Regression and Linear SVM are distance/scale-sensitive linear
    models, so they are wrapped in a Pipeline with StandardScaler.  Random
    Forests split on feature thresholds and do not require scaling.
    """

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_classifier(
    model: Pipeline | RandomForestClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str,
    output_dir: Path,
    models_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    """Fit one classifier, compute metrics, save model and confusion matrix."""

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    labels = np.arange(len(label_encoder.classes_))
    label_names = label_encoder.classes_.tolist()
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report_labels = sorted(np.union1d(y_test, y_pred).tolist())
    report_label_names = label_encoder.inverse_transform(report_labels).tolist()

    figure_path = figures_dir / f"confusion_matrix_{model_name}.png"
    save_confusion_matrix_figure(cm, label_names, model_name, figure_path)

    model_path = models_dir / f"{model_name}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=report_labels,
        target_names=report_label_names,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        labels=report_labels,
        target_names=report_label_names,
        zero_division=0,
    )

    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(
            f1_score(
                y_test,
                y_pred,
                labels=report_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm.tolist(),
        "model_path": str(model_path),
        "confusion_matrix_figure": str(figure_path),
    }


def save_confusion_matrix_figure(
    confusion_values: np.ndarray,
    label_names: list[str],
    model_name: str,
    output_path: Path,
) -> None:
    """Save a readable confusion matrix figure for one model."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_values,
        display_labels=label_names,
    )
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix: {model_name.replace('_', ' ').title()}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    split_info: SplitInfo,
    label_encoder: LabelEncoder,
) -> None:
    """Save reproducibility details for the experiment."""

    metadata = {
        "csv_path": str(args.csv_path),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "random_state": RANDOM_STATE,
        "test_size": args.test_size,
        "labels": label_encoder.classes_.tolist(),
        "split_info": asdict(split_info),
        "methodological_note": (
            "The split is grouped by child_id to avoid speaker leakage. Labels "
            "represented by only one child are kept in training because grouped "
            "splitting cannot place the same child in both train and test."
        ),
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_metrics(output_dir: Path, metrics: list[dict[str, Any]]) -> None:
    """Save metrics as JSON and a compact CSV comparison table."""

    metrics_path = output_dir / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    comparison = pd.DataFrame(
        [
            {
                "model_name": metric["model_name"],
                "accuracy": metric["accuracy"],
                "macro_f1": metric["macro_f1"],
                "model_path": metric["model_path"],
                "confusion_matrix_figure": metric["confusion_matrix_figure"],
            }
            for metric in metrics
        ]
    ).sort_values("macro_f1", ascending=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)


def main() -> None:
    """Run the complete training and evaluation workflow."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(args.csv_path)

    print("Creating grouped train/test split by child_id...")
    train_df, test_df, split_info = grouped_train_test_split(
        df,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
    )
    print(split_info)

    print(f"Encoding text with {EMBEDDING_MODEL_NAME}...")
    x_train, x_test = encode_texts(
        train_df["text"].tolist(),
        test_df["text"].tolist(),
        batch_size=args.batch_size,
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["l1"])
    y_test = label_encoder.transform(test_df["l1"])
    joblib.dump(label_encoder, args.models_dir / "label_encoder.joblib")

    metrics: list[dict[str, Any]] = []
    for model_name, classifier in build_classifiers().items():
        print(f"\nTraining {model_name}...")
        metric = evaluate_classifier(
            model=classifier,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            label_encoder=label_encoder,
            model_name=model_name,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
            figures_dir=args.figures_dir,
        )
        metrics.append(metric)
        print(f"Accuracy: {metric['accuracy']:.3f}")
        print(f"Macro F1: {metric['macro_f1']:.3f}")

    save_metrics(args.output_dir, metrics)
    save_run_metadata(args.output_dir, args, split_info, label_encoder)

    best = max(metrics, key=lambda item: item["macro_f1"])
    print("\nFinished training.")
    print(f"Best model by macro F1: {best['model_name']} ({best['macro_f1']:.3f})")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
