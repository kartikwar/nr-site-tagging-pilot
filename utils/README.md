# utils/

This folder contains all modular helper scripts used by the main document classification pipeline.

---

## Module Descriptions

### `__init__.py`
Marks this folder as a Python package. Can be left empty.

---

### `classifier.py`
- Classifies documents by type (e.g., REPORT, PSI, CORR).
- Supports two modes:
  - `regex`: Uses keyword heuristics on the filename or title.
  - `ml`: Uses a Hugging Face fine-tuned BERT model for classification (optional).
- Falls back to regex if the ML model is unavailable or not selected.

---

### `file_organizer.py`
- Copies the document to `data/output/{DOC_TYPE}/` using a clean filename.
- Automatically creates nested folders for each document type if needed.

---

### `gold_data_extraction.py`
- Loads rows from the official `clean_metadata.csv` file provided by the partner.
- Used to retrieve "gold standard" labels for each document to support evaluation.
- Parses only rows starting from header line 4 to skip metadata instructions.

---

### `llm_interface.py`
- Interfaces with Ollama to run local quantized LLMs (LLaMA 2 or Mistral).
- Sends structured few-shot prompts to extract metadata (site ID, title, etc.).
- Includes support for fallback prompts to recover individual fields.
- Offers functions for both full-record and single-field extraction.

---

### `loader.py`
- Loads `.pdf` files from the `data/input/` directory.
- Uses PyMuPDF to extract and optionally clean OCR text.
- Allows capping the number of pages extracted to improve performance.

---

### `logger.py`
- Writes metadata to `data/logs/metadata_log.csv`.
- Creates the CSV if it doesn’t exist and appends new rows as the pipeline runs.

---

### `metadata_extractor.py`
- Contains helper functions for parsing and validating extracted metadata.
- Includes logic for detecting site IDs in filenames using regex patterns.

---

### `rename.py`
- Generates a standardized filename using extracted metadata:
  `YYYY-MM-DD – SITE_ID – DOC_TYPE.pdf`
- Uses classification output from `classifier.py` to embed document type.
- Ignores case and special characters when parsing original names.