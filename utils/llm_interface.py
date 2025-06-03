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
            "readable": "no"
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

def validate_and_reprompt_field(field_name, length, reprompt_path, metadata_dict, text, filename, flagged_for_review, max_retries=5):
    """
    Validates a metadata field and attempts to re-prompt it up to `max_retries` times if not well-formed.
    Flags the field for manual review if still not valid after retries.

    field_name (str): the metadata_dict key whose value is to be verified/modified.
    length (int): the maximum acceptable length for the field.
    reprompt_path (Path): the path of the prompt template for the field to be re-prompted.
    metadata_dict (dict): the full metadata_dict object being verified.
    text (str): the full cleaned OCR text.
    filename (str): the full filename.
    flagged_for_review (defaultdict): the dictionary containing flags for unverifiable fields.
    max_retries (int): the maximum number of re-prompting attempts before giving up/proceeding.
    """
    field_value = metadata_dict[field_name].strip().lower()
    if field_value != 'none':
        retries = 0
        while (
            metadata_dict[field_name].strip().lower() != 'none' and
            not field_is_well_formed(metadata_dict[field_name], text, length=length) and
            retries < max_retries
        ):
            print(f"Retrying {field_name} extraction, attempt {retries + 1}/{max_retries}")
            reprompt = load_prompt_template(reprompt_path, text)
            metadata_dict[field_name] = llm_single_field_query(reprompt, model="mistral")
            retries += 1

        if metadata_dict[field_name].strip().lower() != 'none' and not field_is_well_formed(metadata_dict[field_name], text, length=length):
            flagged_for_review[filename].append(field_name)
            print(f"{filename} flagged for manual review: {field_name.upper()}")

def keys_are_well_formed(metadata_dict):
    """
    Validates that all expected keys, and only expected keys, are presented in metadata_dict.
    
    metadata_dict (dict): the metadata_dict object being verified."""

    if metadata_dict.keys() == {
    "site_id",
    "title",
    "receiver",
    "sender",
    "address",
    "readable"
    }:
        return True
    return False