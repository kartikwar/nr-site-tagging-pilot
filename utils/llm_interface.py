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



def all_words_in_text(field, text):
    """A simple function to check whether the words in the extracted field are indeed all 
    present in the document. Ignores word order. Used to verify the LLM is not hallucinating.
    
    field (str): the field (title, sender, receiver, etc) extracted by the LLM.
    text (str): the OCR-cleaned text from which the title was extracted."""

    # Remove all punctuation and digits before checking (LLM occasionally adds harmless punctuation)
    field = re.sub(r'[^\w\s]', ' ', field)
    field = re.sub(r'\d+', ' ', field)

    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    # If field is nothing but whitespace, punctuation, or digits, reject it
    if not field.strip():
        return False

    # If any word in field does not occur in actual text, reject it
    for word in field.lower().split():
        if word not in text.lower().split():
            return False
    return True

def field_is_well_formed(field, text, length):
    """Check if title is an appropriate length, and all words in title appear in text.

    field (str): the field (title, sender, receiver, etc) extracted by the LLM.
    text (str): the OCR-cleaned text from which the title was extracted.
    length (int): the desired length of the field, in tokens.
    """
    if len(field.split()) < length and all_words_in_text(field, text):
        return True
    return False