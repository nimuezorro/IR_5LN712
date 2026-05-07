#!/usr/bin/env python3
"""Build a text classification dataset from CHILDES/CHAT bilingual transcripts.

This script extracts the target child's English utterances from `.cha` files and
turns them into medium-length text chunks for supervised machine learning.  The
label is the child's first language (L1).

The important design choice is that utterances are read with `pylangacq`, which
understands CHAT syntax.  That is safer than treating CHAT as plain text because
CHAT contains participant tiers, dependent tiers, retracing markers, omissions,
phonological forms, comments, and other annotation conventions.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pylangacq
from datasets import Dataset, DatasetDict


SCRIPT_DIR = Path(__file__).resolve().parent

# CHAT/PyLangAcq may leave a few transcription artefacts in the readable string.
# The cleaning function below keeps normal English word forms and contractions,
# but removes punctuation, CHAT symbols, unintelligible material, and bare codes.
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
UNINFORMATIVE_TOKENS = {
    "xxx",  # unintelligible speech in CHAT
    "yyy",  # phonologically unclear material in CHAT
    "www",  # untranscribed material in CHAT
    "0",  # omitted/non-verbal response in CHAT
}


def parse_args() -> argparse.Namespace:
    """Read command-line arguments.

    Defaults are set for the current assignment folder, but all paths can be
    overridden so the same script can be reused with another CHILDES dataset.
    """

    parser = argparse.ArgumentParser(
        description="Create an L1 classification dataset from CHILDES .cha files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR / "data/archive (1)",
        help="Directory containing .cha files. Searched recursively.",
    )
    parser.add_argument(
        "--guide-csv",
        type=Path,
        default=SCRIPT_DIR / "data/archive (1)/guide_to_files.csv",
        help="Optional CSV with file-level metadata such as first_language.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "output/childes_l1_dataset",
        help="Directory where CSV, JSONL, and Hugging Face outputs are saved.",
    )
    parser.add_argument(
        "--min-utterance-words",
        type=int,
        default=3,
        help="Drop individual child utterances shorter than this many words.",
    )
    parser.add_argument(
        "--min-chunk-words",
        type=int,
        default=30,
        help="Preferred minimum number of words per ML sample.",
    )
    parser.add_argument(
        "--max-chunk-words",
        type=int,
        default=80,
        help="Preferred maximum number of words per ML sample.",
    )
    return parser.parse_args()


def load_guide_metadata(guide_csv: Path) -> dict[str, dict[str, Any]]:
    """Load optional file-level metadata from `guide_to_files.csv`.

    The guide supplied with this corpus contains full language names such as
    "Spanish" and "Mandarin", while CHAT headers often contain ISO-style codes
    such as "spa" or "cmn".  When both sources are present, the guide is used as
    a human-readable override for L1.
    """

    if not guide_csv.exists():
        return {}

    guide = pd.read_csv(guide_csv)
    guide.columns = [column.strip() for column in guide.columns]

    metadata: dict[str, dict[str, Any]] = {}
    for row in guide.to_dict(orient="records"):
        raw_name = row.get("file_name")
        if pd.isna(raw_name):
            continue
        filename = str(raw_name).strip()
        metadata[filename] = {
            "l1": clean_metadata_value(row.get("first_language")),
            "age": clean_metadata_value(row.get("age_at_recording_months")),
        }
    return metadata


def clean_metadata_value(value: Any) -> str | None:
    """Normalize missing values from headers/CSV to real Python None."""

    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        # Some header values may be list-like or object-like.  Those are not
        # missing values; they simply need to be stringified below if used.
        pass
    text = str(value).strip()
    return text if text else None


def first_existing_header_value(headers: dict[str, Any], possible_keys: list[str]) -> Any:
    """Return a header value even if capitalization differs across parsers."""

    normalized = {str(key).lower(): value for key, value in headers.items()}
    for key in possible_keys:
        value = normalized.get(key.lower())
        if value not in (None, ""):
            return value
    return None


def header_to_dict(raw_headers: Any) -> dict[str, Any]:
    """Convert PyLangAcq header objects to a plain dictionary when possible.

    Different PyLangAcq versions have represented headers slightly differently.
    The current versions return mapping-like objects, while older versions return
    dictionaries.  Keeping this adapter small makes the rest of the code easier
    to read and more robust for university/research environments.
    """

    if isinstance(raw_headers, dict):
        return raw_headers

    if hasattr(raw_headers, "items"):
        return dict(raw_headers.items())

    if hasattr(raw_headers, "__dict__"):
        return vars(raw_headers)

    return {}


def extract_child_id_from_headers(headers: dict[str, Any]) -> str | None:
    """Extract the target child code, for example `BRND`, from CHAT headers."""

    participants = first_existing_header_value(headers, ["Participants"])
    if isinstance(participants, dict):
        child = participants.get("CHI")
        if isinstance(child, dict):
            return clean_metadata_value(child.get("name") or child.get("id"))
        return clean_metadata_value(child)

    # Typical raw CHAT header:
    # @Participants: CHI BRND Target_Child , EXP Lindsay Investigator
    if participants:
        match = re.search(r"\bCHI\s+([^\s,]+)", str(participants))
        if match:
            return match.group(1)

    return None


def extract_l1_from_headers(headers: dict[str, Any]) -> str | None:
    """Extract L1 from headers such as `@L1 of CHI: spa` when available."""

    direct_l1 = first_existing_header_value(
        headers,
        [
            "L1 of CHI",
            "L1",
            "First language",
            "first_language",
        ],
    )
    if direct_l1:
        return clean_metadata_value(direct_l1)

    return None


def extract_age_from_chat(chat: Any, headers: dict[str, Any]) -> str | None:
    """Extract child age, preferring PyLangAcq's age parser when available."""

    try:
        ages = chat.ages()
    except Exception:
        ages = []

    if ages:
        age = ages[0]
        if age is None:
            return None
        if hasattr(age, "in_months"):
            return str(age.in_months())
        return str(age)

    # Fallback for headers with raw @ID lines.  The CHI age is usually the
    # fourth pipe-separated field, e.g. eng|Paradis|CHI|5;05.23|male|...
    ids = first_existing_header_value(headers, ["ID"])
    if not ids:
        return None

    if not isinstance(ids, list):
        ids = [ids]

    for line in ids:
        fields = str(line).split("|")
        if len(fields) > 3 and fields[2] == "CHI":
            return clean_metadata_value(fields[3])
    return None


def read_file_metadata(
    cha_file: Path,
    chat: Any,
    guide_metadata: dict[str, dict[str, Any]],
) -> dict[str, str | None]:
    """Collect metadata for one transcript, tolerating missing fields."""

    try:
        raw_headers = chat.headers()[0]
    except Exception:
        raw_headers = {}

    headers = header_to_dict(raw_headers)
    guide_row = guide_metadata.get(cha_file.name, {})

    child_id = extract_child_id_from_headers(headers) or cha_file.stem.upper()
    l1 = guide_row.get("l1") or extract_l1_from_headers(headers)
    age = guide_row.get("age") or extract_age_from_chat(chat, headers)

    return {
        "child_id": child_id,
        "l1": clean_metadata_value(l1),
        "age": clean_metadata_value(age),
        "source_file": str(cha_file),
    }


def utterance_to_text(utterance: Any) -> str:
    """Get the cleanest readable text exposed by PyLangAcq for an utterance."""

    # PyLangAcq 0.23 uses `audible`; some older releases used `raw`.
    for attribute in ("audible", "raw"):
        value = getattr(utterance, attribute, None)
        if value:
            return str(value)

    # Final fallback: use the original main speaker tier if available.
    tiers = getattr(utterance, "tiers", {}) or {}
    if "CHI" in tiers:
        return str(tiers["CHI"])

    # Token fallback keeps the script usable across versions and unusual data.
    tokens = getattr(utterance, "tokens", []) or []
    words = [getattr(token, "word", "") for token in tokens]
    return " ".join(word for word in words if word)


def clean_utterance(text: str) -> str:
    """Lowercase and remove CHAT artefacts, symbols, and empty material."""

    text = text.lower()
    words = WORD_RE.findall(text)
    words = [word for word in words if word not in UNINFORMATIVE_TOKENS]
    return " ".join(words)


def extract_child_utterances(
    chat: Any,
    min_utterance_words: int,
) -> list[str]:
    """Return cleaned CHI utterances from an already parsed CHAT object."""

    # Filtering by participant is provided by PyLangAcq and avoids accidentally
    # mixing investigator or parent speech into the classification features.
    child_chat = chat.filter(participants="CHI")

    cleaned: list[str] = []
    for utterance in child_chat.utterances():
        text = clean_utterance(utterance_to_text(utterance))
        if len(text.split()) >= min_utterance_words:
            cleaned.append(text)
    return cleaned


def chunk_utterances(
    utterances: list[str],
    min_words: int,
    max_words: int,
) -> list[str]:
    """Combine consecutive child utterances into approximately 30-80 word chunks.

    Consecutive utterances preserve local discourse context, which is useful for
    classification.  Chunks that never reach `min_words` are dropped because very
    short samples tend to be noisy and weak for model training.
    """

    chunks: list[str] = []
    current_words: list[str] = []

    for utterance in utterances:
        utterance_words = utterance.split()

        would_exceed_max = len(current_words) + len(utterance_words) > max_words
        current_is_large_enough = len(current_words) >= min_words

        if current_words and would_exceed_max and current_is_large_enough:
            chunks.append(" ".join(current_words))
            current_words = []

        current_words.extend(utterance_words)

    if len(current_words) >= min_words:
        chunks.append(" ".join(current_words))

    return chunks


def build_dataframe(
    data_dir: Path,
    guide_csv: Path,
    min_utterance_words: int,
    min_chunk_words: int,
    max_chunk_words: int,
) -> pd.DataFrame:
    """Create the final pandas DataFrame from all CHAT files."""

    guide_metadata = load_guide_metadata(guide_csv)
    rows: list[dict[str, str | None]] = []

    cha_files = sorted(data_dir.rglob("*.cha"))
    if not cha_files:
        raise FileNotFoundError(f"No .cha files found under: {data_dir}")

    for cha_file in cha_files:
        chat = pylangacq.read_chat(str(cha_file), strict=False)
        metadata = read_file_metadata(cha_file, chat, guide_metadata)
        utterances = extract_child_utterances(chat, min_utterance_words)
        chunks = chunk_utterances(utterances, min_chunk_words, max_chunk_words)

        for chunk in chunks:
            rows.append(
                {
                    "text": chunk,
                    "l1": metadata["l1"],
                    "child_id": metadata["child_id"],
                    "age": metadata["age"],
                    "source_file": (metadata["source_file"].split('/')[-1]),
                }
            )

    df = pd.DataFrame(rows, columns=["text", "l1", "child_id", "age", "source_file"])

    # Classification labels must be present.  Rows without L1 cannot be used for
    # supervised learning, but keeping this filtering late makes auditing easier.
    df = df.dropna(subset=["l1"])
    df = df[df["l1"].astype(str).str.strip() != ""].reset_index(drop=True)
    return df


def print_statistics(df: pd.DataFrame) -> None:
    """Print high-level dataset and class distribution statistics."""

    print("\nDataset summary")
    print("===============")
    print(f"Rows/samples: {len(df)}")
    print(f"Unique children: {df['child_id'].nunique()}")
    print(f"Unique L1 labels: {df['l1'].nunique()}")

    print("\nClass distribution")
    print("==================")
    print(df["l1"].value_counts(dropna=False).to_string())

    print("\nMean words per sample by L1")
    print("===========================")
    word_counts = df["text"].str.split().str.len()
    print(word_counts.groupby(df["l1"]).mean().round(1).to_string())


def save_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    """Save CSV, JSONL, and Hugging Face datasets-compatible outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "childes_l1_dataset.csv"
    jsonl_path = output_dir / "childes_l1_dataset.jsonl"
    hf_path = output_dir / "hf_dataset"

    df.to_csv(csv_path, index=False)

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for row in df.to_dict(orient="records"):
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Hugging Face `datasets` can load this folder with:
    #   from datasets import load_from_disk
    #   dataset = load_from_disk("output/childes_l1_dataset/hf_dataset")
    dataset = Dataset.from_pandas(df, preserve_index=False)
    DatasetDict({"train": dataset}).save_to_disk(hf_path)

    print("\nSaved outputs")
    print("=============")
    print(f"CSV: {csv_path}")
    print(f"JSONL: {jsonl_path}")
    print(f"Hugging Face dataset: {hf_path}")


def main() -> None:
    """Run the full preprocessing pipeline."""

    args = parse_args()

    if args.min_chunk_words > args.max_chunk_words:
        raise ValueError("--min-chunk-words must be <= --max-chunk-words")

    df = build_dataframe(
        data_dir=args.data_dir,
        guide_csv=args.guide_csv,
        min_utterance_words=args.min_utterance_words,
        min_chunk_words=args.min_chunk_words,
        max_chunk_words=args.max_chunk_words,
    )
    print_statistics(df)
    save_outputs(df, args.output_dir)


if __name__ == "__main__":
    main()
