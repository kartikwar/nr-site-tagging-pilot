import csv
from pathlib import Path

_log_headers = []

def init_log(filepath: Path, headers: list):
    """
    Initializes a CSV log file with specified column headers.
    Creates parent directories if they do not exist.
    """
    global _log_headers
    _log_headers = headers
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

def log_metadata(filepath: Path, row: dict):
    """
    Appends a row of metadata to the existing CSV log file using the original header order.
    Missing keys are filled with an empty string.
    """
    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([row.get(col, "") for col in _log_headers])