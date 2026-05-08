# Legacy Hugging Face Model Export

The canonical trained artifacts now live in `../models/`.

This folder is kept for compatibility with earlier upload commands. The model
files here are copies of the deployed Random Forest classifier and label encoder.
New code should load paths from `src.paths` instead of relying on the current
working directory.
