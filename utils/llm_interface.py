import ollama
import re


def load_prompt_template(path, doc_text: str) -> str:
    """
    Loads a text prompt template from file and injects document text into the placeholder.

    Parameters:
        path (str or Path): Path to the prompt template file.
        doc_text (str): The cleaned OCR text to insert into the prompt.

    Returns:
        str: A complete prompt with the document text inserted into {{DOCUMENT_TEXT}}.
    """
    with open(path, "r") as file:
        template = file.read()
    return template.replace("{{DOCUMENT_TEXT}}", doc_text.strip()[:3000])


def query_llm(prompt, model="llama2", system_prompt=None):
    """
    Sends a prompt to a locally running Ollama LLM (e.g., LLaMA 2 or Mistral) and returns a metadata dictionary.

    Parameters:
        prompt (str): The main user prompt containing the document and extraction instructions.
        model (str): The name of the local Ollama model to query (default: "llama2").
        system_prompt (str): Optional system-level prompt for context or instruction.

    Returns:
        dict: A dictionary with keys like site_id, title, receiver, sender, address, and readable.
              If the model response fails or is invalid, returns a default dictionary with "none" values.
    """
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    try:
        response = ollama.chat(model=model, messages=messages)
        raw = response['message']['content'].strip()
        metadata_dict = eval(raw)
          
    except Exception as e:
        print(e)
        metadata_dict = {
            "site_id": "none",
            "title": "none",
            "receiver": "none",
            "sender": "none",
            "address": "none",
            "readable": "none"
        }
        
    return metadata_dict


def llm_single_field_query(prompt, model="llama2", system_prompt=None) -> str:
    """
    Sends a prompt to the LLM and returns a single field value (as string) instead of a dictionary.
    Used when retrying extraction of individual metadata fields.

    Parameters:
        prompt (str): A targeted prompt designed to extract a specific field (e.g., site_id or address).
        model (str): Ollama model name to query (default: "llama2").
        system_prompt (str): Optional system-level prompt to include.

    Returns:
        str: The raw string value returned by the model for the requested field.
    """
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    response = ollama.chat(model=model, messages=messages)
    raw = response['message']['content'].strip()

    return raw
