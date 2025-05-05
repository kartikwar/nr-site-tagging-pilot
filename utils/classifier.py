import re

def classify_document(file_path, metadata=None):
    """Classify document type based on the renamed file's name."""
    filename = file_path.stem.upper()  # Get filename without .pdf
    doc_type_match = re.search(r"\b(CORR|REPORT|NIR|DSI|PSI|HHERA|COR|COC|AIP|DET)\b", filename)

    if doc_type_match:
        doc_type = doc_type_match.group(1)
    else:
        doc_type = "UNKNOWN"

    print(f"[Classifier] {file_path.name} classified as {doc_type}")
    return doc_type