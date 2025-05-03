from pathlib import Path
import os
import fitz  # PyMuPDF

def load_pdfs(pdf_dir: Path):
    """Return a list of all PDF files in the given directory."""
    return sorted([file for file in pdf_dir.glob("*.pdf") if file.is_file()])

def extract_text_from_pdf(pdf_path):
    """Extract full text from a PDF using PyMuPDF."""
    with fitz.open(pdf_path) as doc:
        return "\n".join([page.get_text() for page in doc])