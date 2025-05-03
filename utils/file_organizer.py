import shutil
from pathlib import Path

def organize_files(original_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(original_path, output_path)
    print(f"[Organize] Copied to {output_path}")