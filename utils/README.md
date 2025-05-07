# utils/

This folder contains all modular helper scripts used by the main document classification pipeline.

---

## Module Descriptions

### `__init__.py`
Marks this folder as a Python package. Can be left empty.

---

### `classifier.py`
- Classifies a document’s type (e.g., REPORT, PSI, CORR).
- Uses regex-based keyword matching on filenames and text.
- Future versions may include trained classifiers.

---

### `file_organizer.py`
- Copies a file into `outputs/{DOC_TYPE}/` using its new standardized filename.
- Ensures output folders are created automatically.

---

### `llm_interface.py`
- Interfaces with Ollama to send prompts to a quantized LLaMA 2 model running locally.
- Used to extract the site ID if it’s missing from the filename.
- Uses few-shot prompting for structured, predictable output.

---

### `loader.py`
- Loads `.pdf` files from the `data/` directory (located one level up from the repo).
- Uses PyMuPDF to extract text content from each PDF.
- Returns file paths and extracted text for downstream processing.

---

### `logger.py`
- Handles logging to a central CSV file: `logs/metadata_log.csv`.
- Logs original filename, new filename, site ID, document type, and output path.
- Supports creating the CSV file if it doesn’t already exist.

---

### `metadata_rules.py`
- Placeholder for future rule-based metadata extraction logic.
- Intended to extract structured fields like `address`, `sender`, and `receiver`.
- Currently returns hardcoded placeholders (`"Unknown"` or `"N/A"`).

---

### `rename.py`
- Generates a new standardized filename using date, site ID, and document type.
- Format: `YYYY-MM-DD – SITE_ID – TYPE.pdf`
- Does **not** modify the original file — just returns the new filename string.
