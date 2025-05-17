import ollama
import re
from difflib import SequenceMatcher

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



def all_words_in_text(title, text):
    """A simple function to check whether the words in the extracted title are indeed all 
    present in the document. Ignores word order. Used to verify the LLM is not hallucinating.
    
    title (str): the title extracted by the LLM.
    text (str): the OCR-cleaned text from which the title was extracted."""

    # Remove all punctuation and digits before checking
    title = re.sub(r'[^\w\s]', ' ', title)
    title = re.sub(r'\d+', ' ', title)

    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    for word in title.lower().split():
        if word not in text.lower().split():
            return False
    return True

def title_is_well_formed(title, text):
    """Check if title is an appropriate length, and all words in title appear in text.

    title (str): the title extracted by the LLM.
    text (str): the OCR-cleaned text from which the title was extracted.
    """
    if len(title.split()) < 22 and all_words_in_text(title, text):
        return True
    return False