from pathlib import Path
import os
import fitz  # PyMuPDF
import re

def load_pdfs(pdf_dir: Path):
    """Return a list of all PDF files in the given directory."""
    return sorted([file for file in pdf_dir.glob("*.pdf") if file.is_file()])

def extract_text_from_pdf(pdf_path):
    """Extract full text from a PDF using PyMuPDF."""
    with fitz.open(pdf_path) as doc:
        return "\n".join([page.get_text() for page in doc])

def extract_text_from_pdf(pdf_path, max_pages=5):

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc[:max_pages]:
        text += page.get_text()
    doc.close()
    return text

def clean_ocr_text(text):
   
    text = text.replace("\n", " ")
    text = re.sub(r'[^a-zA-Z0-9\s:,\-./]', '', text)  
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()