# CHILDES L1 Classification Dataset Pipeline

This pipeline creates a clean text classification dataset from CHILDES bilingual
children's speech transcripts in CHAT `.cha` format. The machine-learning task is
to predict a child's first language (L1) from their English utterances.

## How It Works

1. The script recursively finds all `.cha` files under the dataset directory.
2. Each file is opened with `pylangacq`, a library that understands CHAT syntax.
3. Only `CHI` participant utterances are kept, so investigator and parent speech
   are not used as model input.
4. File metadata is collected from CHAT headers and, when available, from
   `guide_to_files.csv`. The guide is used to prefer readable L1 labels such as
   `Spanish` over shorter header codes such as `spa`.
5. Utterances are lowercased and cleaned by removing CHAT artefacts, special
   symbols, punctuation, empty lines, and utterances shorter than three words.
6. Consecutive child utterances are combined into chunks of roughly 30-80 words.
   This gives each classification example more linguistic context than a single
   short child utterance.
7. Rows without an L1 label are removed because they cannot be used for
   supervised classification.
8. The final dataset is saved as CSV, JSONL, and a Hugging Face `datasets`
   folder that can be loaded with `load_from_disk`.

## Usage

From the `assignment_1` directory:

```bash
pip install -r requirements.txt
python build_childes_l1_dataset.py
```

The default input path is:

```text
data/archive (1)
```

The default output path is:

```text
output/childes_l1_dataset
```

To use custom paths:

```bash
python build_childes_l1_dataset.py \
  --data-dir "data/archive (1)" \
  --guide-csv "data/archive (1)/guide_to_files.csv" \
  --output-dir "output/childes_l1_dataset"
```

The resulting table has these columns:

- `text`
- `l1`
- `child_id`
- `age`
- `source_file`

## Training Classifiers

After creating the cleaned CSV, train the L1 classifiers with:

```bash
python train_childes_l1_models.py
```

The training script loads:

```text
output/childes_l1_dataset/childes_l1_dataset.csv
```

It encodes each `text` sample with:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Then it trains and compares:

- Logistic Regression
- Linear SVM
- Random Forest

The train/test split is grouped by `child_id`. This is important because the
dataset contains multiple text chunks from the same child. If chunks from one
child appeared in both train and test, the evaluation could be inflated by
speaker-specific information rather than true generalization to unseen children.

Some L1 labels are represented by only one child. A grouped split cannot place
the same child in both train and test, so these singleton-child labels are kept
in the training set and documented in the run metadata.

The script evaluates each classifier with:

- accuracy
- macro F1
- confusion matrix
- classification report

Training outputs are saved by default to:

```text
output/childes_l1_models
```

This folder contains:

- `models/*.joblib`: trained classifiers and the label encoder
- `evaluation_metrics.json`: full metrics, classification reports, and confusion
  matrix values
- `model_comparison.csv`: compact accuracy/macro-F1 comparison table
- `figures/confusion_matrix_*.png`: confusion matrix figure for each classifier
- `run_metadata.json`: dataset path, embedding model, labels, split details, and
  methodological notes

To use custom paths:

```bash
python train_childes_l1_models.py \
  --csv-path "output/childes_l1_dataset/childes_l1_dataset.csv" \
  --output-dir "output/childes_l1_models"
```
