import re

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