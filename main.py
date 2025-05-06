import torch
import re
from pathlib import Path
from utils.loader import load_pdfs, extract_text_from_pdf, clean_ocr_text
from utils.rename import generate_new_filename
from utils.classifier import classify_document
from utils.file_organizer import organize_files
from utils.llm_interface import query_llm, load_prompt_template
from utils.logger import init_log, log_metadata
from utils.metadata_extractor import extract_site_id_from_filename
import config



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
    prompt_path = Path("prompts/metadata_prompt.txt")

    print(f"Scanning directory: {input_dir.resolve()}")
    files = load_pdfs(input_dir)

    # init_log(log_path, headers=[
    #     "original_filename", "new_filename", "site_id", "document_type", "title", 
    #     "receiver", "sender", "address", "readable", "output_path"
    # ])

    if not files:
        print("No PDF files found.")
        return

    for file_path in files:
        filename = file_path.name
        site_id = extract_site_id_from_filename(filename)

        if site_id:
            print(f"[Extracted from filename] Site ID: {site_id}")
        else:
            print("[Fallback to LLM] Site ID not found in filename")

        # Extract only first 5 pages of text
        text = extract_text_from_pdf(file_path, max_pages=5)
        
        prompt = load_prompt_template(prompt_path,  clean_ocr_text(text))
        
        metadata_dict = query_llm(prompt, model="mistral")

        # Extract site id values
        llm_site_id = metadata_dict.get("site_id", "none")

        # Compare and decide which to use
        if site_id and llm_site_id != "none" and site_id != llm_site_id:
            print(f"[Site ID MISMATCH] Filename: {site_id}, LLM: {llm_site_id} — using filename version")
        elif not site_id and llm_site_id != "none":
            site_id = llm_site_id
            print(f"[Site ID FROM LLM] {site_id}")
        elif not site_id:
            site_id = "UNKNOWN"

            
        #new_filename = generate_new_filename(file_path, site_id=site_id)
        doc_type = classify_document(file_path, {"site_id": site_id})
        #output_path = output_dir / doc_type / new_filename

        print("metadata response: ", metadata_dict)
        print("final site id: ", site_id)
        
        #organize_files(file_path, output_path)
        # log_metadata(log_path, {
        #     "original_filename": file_path.name,
        #     "new_filename": new_filename,
        #     "site_id": site_id,
        #     "document_type": doc_type,
        #     "title": metadata_dict.get("title", "none"),
        #     "receiver": metadata_dict.get("receiver", "none"),
        #     "sender": metadata_dict.get("sender", "none"),
        #     "address": metadata_dict.get("address", "none"),
        #     "readable": metadata_dict.get("readable", "no"),
        #     "output_path": str(output_path)
        # })


    print("Pipeline complete.")

if __name__ == "__main__":
    main()