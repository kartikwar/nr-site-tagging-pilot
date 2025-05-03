from pathlib import Path

# External data directory
PDF_DATA_PATH = Path(__file__).resolve().parents[1] / "data"

INPUT_DIR = PDF_DATA_PATH / "input"
OUTPUT_DIR = PDF_DATA_PATH / "output"
LOG_PATH = PDF_DATA_PATH / "logs" / "metadata_log.csv"