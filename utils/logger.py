import csv
from pathlib import Path

def init_log(filepath: Path, headers: list):
    """
    Initializes a CSV log file with specified column headers.
    Creates parent directories if they do not exist.

    Parameters:
        filepath (Path): Full path to the CSV log file.
        headers (list): List of column names to write as the first row.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists():
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def log_metadata(filepath: Path, row: dict):
    """
    Appends a row of metadata to the existing CSV log file.

    Parameters:
        filepath (Path): Full path to the CSV log file.
        row (dict): Dictionary of metadata fields to log, with keys matching the CSV headers.
    """
    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([row[key] for key in row])
