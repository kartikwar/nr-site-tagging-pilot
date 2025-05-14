import re
import os
from pathlib import Path
from .loader import extract_text_from_pdf, clean_ocr_text
from rouge_score import rouge_scorer
import pandas as pd
import sys

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


def check_duplicate_by_rouge(current_text, site_id_dir: Path, threshold=0.8, rouge_metric="rouge1"):
    """
    Compares current_text against all PDFs in site_id_dir using ROUGE.
    
    Parameters:
        current_text (str): The cleaned text of the current PDF.
        site_id_dir (Path): The full path to output_dir / site_id.
        threshold (float): Minimum score (precision, recall, or F1) to consider as duplicate.
        rouge_metric (str): 'rouge1', 'rouge2', or 'rougeL'

    Returns:
        'yes' if any file is a duplicate based on the threshold, else 'no'.
    """
    scorer = rouge_scorer.RougeScorer([rouge_metric], use_stemmer=True)

    # Skip if the folder doesn't exist yet
    if not site_id_dir.exists():
        return "no"

    for root, _, files in os.walk(site_id_dir):
        for file in files:
            if file.endswith(".pdf"):
                file_path = Path(root) / file
                try:
                    other_text = extract_text_from_pdf(file_path, max_pages=8)
                    other_text = clean_ocr_text(other_text)
                    scores = scorer.score(current_text, other_text).get(rouge_metric, None)

                    if scores:
                        if scores.precision >= threshold or scores.recall >= threshold or scores.fmeasure >= threshold:
                            print(f"[DUPLICATE DETECTED] Compared with {file} — Precision: {scores.precision:.2f}, Recall: {scores.recall:.2f}, F1: {scores.fmeasure:.2f}")
                            return "yes"
                except Exception as e:
                    print(f"[ERROR reading {file}] {e}")
                    continue

    return "no"


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


