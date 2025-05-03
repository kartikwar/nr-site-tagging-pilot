import csv
from pathlib import Path

def init_log(filepath: Path, headers: list):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists():
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def log_metadata(filepath: Path, row: dict):
    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([row[key] for key in row])