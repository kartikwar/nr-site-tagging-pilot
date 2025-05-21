import re
import os
import string
from pathlib import Path
from .loader import extract_text_from_pdf, clean_ocr_text
from rouge_score import rouge_scorer
import pandas as pd
import sys
import fitz
from rapidfuzz import fuzz

def extract_site_id_from_filename(filename):
    """
    Extracts a numeric site ID from the start of a filename using regex.
    
    Parameters:
        filename (str): The name of the file to extract the site ID from.
    
    Returns:
        str or None: A 3- to 5-digit site ID if matched; otherwise, None.
    """
    match = re.match(r"^(\d{3,5})\b", filename)
    if match:
        return match.group(1)
    return None


def extract_metadata(file_path):
    """
    Returns placeholder metadata for testing or debugging purposes.

    Parameters:
        file_path (Path): Path to the input file (used for display only).

    Returns:
        dict: A dictionary containing dummy values for site_id, address, sender, and receiver.
    """
    metadata = {
        "site_id": "12345",
        "address": "N/A",
        "sender": "Unknown",
        "receiver": "Unknown"
    }
    print(f"[Metadata] Extracted from {file_path.name}: {metadata}")
    return metadata

_punct = str.maketrans("", "", string.punctuation)
_space = re.compile(r"\s+")

def _clean(txt: str) -> str:
    return _space.sub(" ", txt.lower().translate(_punct)).strip()

def _best_window_min_f1(shorter_pages, longer_pages, scorer) -> float:
    best = 0.0
    for i in range(len(longer_pages) - len(shorter_pages) + 1):
        window = longer_pages[i:i+len(shorter_pages)]
        worst = min(
            scorer.score(a, b)["rouge1"].fmeasure
            for a, b in zip(shorter_pages, window)
        )
        best = max(best, worst)
    return best

def check_duplicate_by_rouge(
        current_text,
        site_id: str,
        site_id_dir: Path,
        rouge_th: float = 0.75,
        rapid_th: float = 78.0,
        rouge_metric: str = "rouge1",
) -> tuple[str, Path | None, bool, float]:
    """
    Two-step duplicate detector.

    Returns:
        (duplicate_status, matched_file_path, is_current_file_shorter, similarity_score)
    """
    scorer = rouge_scorer.RougeScorer([rouge_metric], use_stemmer=True)

    try:
        cur_doc = fitz.open(current_text) if isinstance(current_text, (str, Path)) else None
    except Exception:
        cur_doc = None

    if cur_doc:
        cur_pages = [_clean(p.get_text()) for p in cur_doc]
        cur_doc.close()
    else:
        cur_pages = [_clean(current_text)]

    if not site_id_dir.exists():
        return "no", None, False, 0.0

    for root, _, files in os.walk(site_id_dir):
        for file in files:
            if not file.lower().endswith(".pdf"):
                continue
            if site_id and str(site_id) not in file:
                continue

            cand_path = Path(root) / file
            try:
                cand_pages = [_clean(p.get_text()) for p in fitz.open(cand_path)]
            except Exception as e:
                print(f"[WARN] {cand_path.name}: {e}")
                continue

            a, b = (cur_pages, cand_pages) if len(cur_pages) <= len(cand_pages) else (cand_pages, cur_pages)
            is_shorter = len(cur_pages) <= len(cand_pages)

            rouge_score = _best_window_min_f1(a, b, scorer)
            if rouge_score >= rouge_th:
                print(f"[CONTAINED] {file}")
                return "contained", cand_path, is_shorter, rouge_score

            rapid_score = fuzz.token_sort_ratio(" ".join(a), " ".join(b))
            if rapid_score >= rapid_th:
                print(f"[LIKELY DUPLICATE (OCR)] {file}")
                return "likely_duplicate_ocr", cand_path, is_shorter, rapid_score / 100.0

    return "no", None, False, 0.0


_release_df = None  # cache for loaded Excel

def get_site_registry_releasable(doc_type: str, lookup_file_path: str) -> str:
    """
    Looks up the Site Registry Releasable status for a given document type.
    If the Excel file is missing or no match is found, the program exits with a message.

    Parameters:
        doc_type (str): Document type (e.g., 'report')
        lookup_file_path (str): Path to Excel lookup file

    Returns:
        str: 'yes' or 'no' based on the lookup file
    """
    global _release_df

    try:
        # Load and cache Excel file if not already
        if _release_df is None:

            _release_df = pd.read_excel(lookup_file_path)
            _release_df['Document_Type'] = _release_df['Document_Type'].astype(str).str.lower()
            _release_df['Site_Registry_Releaseable'] = _release_df['Site_Registry_Releaseable'].astype(str).str.lower().str.strip()

        doc_type = doc_type.lower().strip()

        # Partial (contains) match
        for _, row in _release_df.iterrows():
            if doc_type == row['Document_Type']:
                return row['Site_Registry_Releaseable']

        # No match found — exit safely
        print(f"[ERROR] Document type '{doc_type}' not found in site registry mapping.")
        sys.exit(1)

    except Exception as e:
        print(f"[ERROR] Failed to check site registry releasable: {e}")
        sys.exit(1)


