import torch
import re
from pathlib import Path
from utils.loader import load_pdfs, extract_text_from_pdf, clean_ocr_text
from utils.rename import generate_new_filename
from utils.classifier import classify_document
from utils.file_organizer import organize_files
from utils.llm_interface import query_llm, llm_single_field_query, load_prompt_template
from utils.logger import init_log, log_metadata
from utils.metadata_extractor import extract_site_id_from_filename
from utils.gold_data_extraction import load_gold_data
import config
import ollama
import os


def load_model(device):
    # Optional: load Hugging Face model if ML mode is enabled
    USE_ML_CLASSIFIER = True
    from utils.classifier import load_huggingface_model
    try:
        model_path = os.path.join(
            "models", "document_classification_model")
        load_huggingface_model(model_path, device)
        print("[ML Classifier] Model loaded ")
    except Exception as e:
        print(f"[ML Classifier] Failed to load model: {e}")
        USE_ML_CLASSIFIER = False
    return USE_ML_CLASSIFIER


def main():
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"Using device: {device}")

    USE_ML_CLASSIFIER = load_model(device)
    print(f"value of USE ML {USE_ML_CLASSIFIER}")

    input_dir = config.INPUT_DIR
    output_dir = config.OUTPUT_DIR
    log_path = config.LOG_PATH
    # Main prompt to extract metadata fields
    prompt_path = Path("prompts/metadata_prompt.txt")

    # Additional re-prompts for the LLM, only called if the first pass misses an important field
    # address_reprompt_path = Path("prompts/address_reprompt.txt")
    site_id_reprompt_path = Path("prompts/site_id_reprompt")

    # Initialize a dict object to store successfully retrieved site ID - address pairs.
    # Some documents are missing address, but ground truth address is shared among all docs with same site ID.
    site_id_address_dict = dict()

    print(f"Scanning directory: {input_dir.resolve()}")
    files = load_pdfs(input_dir)

    init_log(log_path, headers=[
        "original_filename", "new_filename", "site_id", "document_type", "title",
        "receiver", "sender", "address", "readable", "output_path"
    ])

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

        # Extract only first 8 pages of text
        text = extract_text_from_pdf(file_path, max_pages=8)

        prompt = load_prompt_template(prompt_path,  clean_ocr_text(text))

        metadata_dict = query_llm(prompt, model="mistral")

        # If title extraction fails, assume metadata extraction has failed entirely. Make up to 5 re-attempts to extract metadata.
        metadata_retries = 0
        while metadata_dict['title'].lower() == 'none' and metadata_retries < 5:
            print(
                f"Retrying metadata extraction, attempt {metadata_retries + 1}/5")
            metadata_dict = query_llm(prompt, model="mistral")
            metadata_retries += 1

        # If extraction of any other required field fails, call LLM to re-attempt just that field
        # while metadata_dict['address'].lower() == 'none':
        #    address_reprompt = load_prompt_template(address_reprompt_path,  clean_ocr_text(text))
        #    print("Retrying ADDRESS extraction...")
        #    metadata_dict['address'] = llm_single_field_query(address_reprompt, model="mistral")

        # Extract site id values only if needed
        llm_site_id = metadata_dict.get("site_id", "none")

        # Only evaluate LLM site ID if filename did not provide a valid one
        if not site_id:
            if not re.fullmatch(r"\d{3,5}", llm_site_id):
                print(f"[Rejected LLM Site ID] {llm_site_id} — invalid format")
                llm_site_id = "none"
            else:
                print(f"[Validated LLM Site ID] {llm_site_id}")
                site_id = llm_site_id
                print(f"[Site ID FROM LLM] {site_id}")

        # Make up to 5 re-attempts to extract site_id
        site_id_retries = 0
        while not site_id and site_id_retries < 5:
            print(
                f"Retrying Site ID extraction, attempt {site_id_retries + 1}/5")
            site_id_reprompt = load_prompt_template(
                site_id_reprompt_path, clean_ocr_text(text))
            proposed_site_id = llm_single_field_query(site_id_reprompt)
            if re.fullmatch(r"\d{3,5}", proposed_site_id):
                site_id = proposed_site_id
                print(f"[Re-prompted Valid Site ID] {site_id}")
                break
            else:
                print(f"[Re-prompted Invalid Site ID] {proposed_site_id}")
                site_id_retries += 1

        # If an address is extracted and no address is recorded for this site ID yet, save it in dict.
        if metadata_dict['address'].lower() != 'none':
            if site_id_address_dict.get(site_id) is None:
                site_id_address_dict[site_id] = metadata_dict['address']

        # If no address is extracted but we have previously extracted an address, re-use it.
        elif site_id_address_dict.get(site_id) is not None:
            print(
                f"Address not found in document. Re-using previously extracted address from site_id: {site_id}")
            metadata_dict['address'] = site_id_address_dict[site_id]

        # Now get document type
        title = metadata_dict.get("title", "").strip()
        if (not title) or (title == 'none') or (USE_ML_CLASSIFIER == False):
            print(f"Using regex mode")
            doc_type = classify_document(file_path, device, {"site_id": site_id, "title": metadata_dict.get(
                "title", "")}, mode="regex")
        else:
            print(f"Using ml mode for {title}")
            doc_type = classify_document(file_path, device, {"site_id": site_id, "title": metadata_dict.get(
                "title", "")}, mode="ml")

        print(f"document type is {doc_type} for {file_path}")

        # Generate filename after doc_type is available
        new_filename = generate_new_filename(
            file_path, site_id=site_id, doc_type=doc_type)

        output_path = output_dir / doc_type / new_filename

        # print("\nmetadata response:\n", metadata_dict)
        # print("final site id: ", site_id)
        # gold_data = load_gold_data(filename, 'clean_metadata.csv')
        # print("\ngold response:\n", gold_data)
        # print('\n----\n')

        organize_files(file_path, output_path)
        log_metadata(log_path, {
            "original_filename": file_path.name,
            "new_filename": new_filename,
            "site_id": site_id,
            "document_type": doc_type,
            "title": metadata_dict.get("title", "none"),
            "receiver": metadata_dict.get("receiver", "none"),
            "sender": metadata_dict.get("sender", "none"),
            "address": metadata_dict.get("address", "none"),
            "readable": metadata_dict.get("readable", "no"),
            "output_path": str(output_path)
        })

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
