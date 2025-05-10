import re
from pathlib import Path

def generate_new_filename(file_path: Path, site_id: str = "UNKNOWN", doc_type: str = "REPORT") -> str:
    filename = file_path.name

    date_match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", filename)
    if date_match:
        date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
    else:
        date_str = "0000-00-00"

    new_name = f"{date_str} – {site_id} – {doc_type.upper()}{file_path.suffix}"
    return new_name