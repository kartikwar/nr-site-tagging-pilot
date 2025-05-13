import re
import os
from pathlib import Path
from .loader import extract_text_from_pdf, clean_ocr_text
from rouge_score import rouge_scorer

def extract_site_id_from_filename(filename):
    match = re.match(r"^(\d{3,5})\b", filename)
    if match:
        return match.group(1)
    return None

def extract_metadata(file_path):
    """Return dummy metadata."""
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