import ollama
import re

def load_prompt_template(path, doc_text: str) -> str:
    with open(path, "r") as file:
        template = file.read()
    return template.replace("{{DOCUMENT_TEXT}}", doc_text.strip()[:3000])

def query_llm(prompt, model="llama2", system_prompt=None):
    """Query LLaMA via Ollama and return a metadata dictionary."""
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
    """Similar to query_llm, but extracts only a single field (str) rather than a dictionary.
    Used when a specific field fails to be extracted by query_llm.
    
    prompt (str): a prompt specific to the field being extracted.
    doc_text (str): full document text. """
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    response = ollama.chat(model=model, messages=messages)
    raw = response['message']['content'].strip()

    return raw