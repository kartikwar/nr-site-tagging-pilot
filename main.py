import torch
import re
from pathlib import Path
from utils.loader import load_pdfs, extract_text_from_pdf
from utils.rename import generate_new_filename
from utils.classifier import classify_document
from utils.file_organizer import organize_files
from utils.llm_interface import query_llm
from utils.logger import init_log, log_metadata
import config

def extract_site_id_from_filename(filename):
    match = re.match(r"^(\d{3,5})\b", filename)
    if match:
        return match.group(1)
    return None

def main():
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    input_dir = config.INPUT_DIR
    output_dir = config.OUTPUT_DIR
    log_path = config.LOG_PATH

    print(f"Scanning directory: {input_dir.resolve()}")
    files = load_pdfs(input_dir)

    init_log(log_path, headers=["original_filename", "new_filename", "site_id", "document_type", "output_path"])

    if not files:
        print("No PDF files found.")
        return

    for file_path in files:
        filename = file_path.name
        site_id = extract_site_id_from_filename(filename)

        if site_id:
            print(f"[Extracted from filename] Site ID: {site_id}")
        else:
            text = extract_text_from_pdf(file_path)
            prompt = (
                "Extract the numeric site ID from the document below. "
                "If no valid site ID is found, return 'UNKNOWN'.\n"
                "Only output the ID number. Do not explain.\n\n"
                "Examples:\n"
                "Text: This report describes work done at Site ID 0141.\nAnswer: 0141\n\n"
                "Text: No site ID is given in this document.\nAnswer: UNKNOWN\n\n"
                f"Text: {text[:3000]}\nAnswer:"
            )

            try:
                site_id = query_llm(prompt, model="llama2")
                print(f"[LLM Site ID] {site_id}")
            except Exception as e:
                site_id = "UNKNOWN"
                print(f"[LLM Site ID ERROR] {e}")

        new_filename = generate_new_filename(file_path, site_id=site_id)
        doc_type = classify_document(file_path, {"site_id": site_id})
        output_path = output_dir / doc_type / new_filename

        metadata = {
            "site_id": site_id,
            "address": "N/A",
            "sender": "Unknown",
            "receiver": "Unknown"
        }

        organize_files(file_path, output_path)
        log_metadata(log_path, {
            "original_filename": file_path.name,
            "new_filename": new_filename,
            "site_id": site_id,
            "document_type": doc_type,
            "output_path": str(output_path)
        })

    print("Pipeline complete.")

if __name__ == "__main__":
    main()