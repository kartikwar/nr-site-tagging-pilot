# utils/

This folder contains all modular helper scripts used by the main document classification pipeline.

---

## Module Descriptions

### `__init__.py`
- Marks this folder as a Python package. Can be left empty.

---

### `classifier.py`
- Classifies a document’s type (e.g., REPORT, PSI, CORR).
- Uses regex-based keyword matching on filenames and/or extracted text.
- May be extended in future versions to use trained models.

---

### `file_organizer.py`
- Copies a file into `data/output/{DOC_TYPE}/` using its new standardized filename.
- Automatically creates output directories if they don’t exist.
- Designed to keep output files separate from the input dataset and repository.

---

### `llm_interface.py`
- Interfaces with Ollama to send prompts to a local quantized LLaMA 2 or Mistral model.
- Used to extract metadata fields like site ID, title, sender, and address.
- Supports both structured (JSON) and single-field prompting.
- Includes re-prompt logic for fallback cases when key fields are missing.

---

### `loader.py`
- Loads `.pdf` files from the input directory (`data/input/`).
- Uses PyMuPDF (`fitz`) to extract and clean text content.
- Also includes OCR cleanup utilities to improve prompt readability.

---

### `logger.py`
- Logs structured metadata for each processed file to `logs/metadata_log.csv`.
- Supports logging fields including:
  - Original and new filename
  - Site ID
  - Document type
  - Title, sender, receiver, address
  - Readability flag
  - Output file path
- Creates the log file and headers if missing.

---

### `metadata_extractor.py`
- Extracts the site ID from a filename using regex.
- Returns a 3–5 digit number if found at the start of the filename.
- Used as the first-pass method before querying the LLM for site ID.

---

### `gold_data_extraction.py`
- Loads gold-standard metadata from `clean_metadata.csv`.
- Matches records using the filename (typically mapped to folder names).
- Used in the pipeline to compare predicted metadata to the official ground truth.

---

### `rename.py`
- Generates standardized output filenames in the format:
  `YYYY-MM-DD – SITE_ID – TYPE.pdf`
- Site ID and document type are injected based on earlier steps.
- Does **not** modify or move the file itself — just returns the new name string.

---

### `metadata_rules.py`
- **[Deprecated]** Placeholder for future rule-based metadata extraction.
- Not currently used in the active pipeline.
