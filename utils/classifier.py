import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Hugging Face model globals
hf_tokenizer = None
hf_model = None
label_encoder = None  # Optional, if using label indices

# Keyword map for regex classification
DOCUMENT_TYPES = {
    "AiP": ["approval in principle", "aip"],
    "CoC": ["certificate of compliance", "coc"],
    "FDet": ["final determination", "fdet"],
    "PDet": ["preliminary determination", "pdet"],
    "PSI": ["preliminary site investigation", "stage 1 psi", "stage 2 psi", "psi"],
    "DSI": ["detailed site investigation", "dsi"],
    "COV": ["covenant", "cov"],
    "CORR": ["correspondence", "letter", "email", "corr"],
    "REPORT": ["remediation", "report", "summary", "investigation"],
}

class_names = [
    'AIP', 'COA', 'COC', 'CORR', 'COV', 'CSSA', 'DSI', 'FDET', 'IMG', 'MAP',
    'NIRI', 'OTHERS', 'PDET', 'PSI', 'RA', 'RPT', 'SP', 'SSI', 'Site Registry',
    'TITLE', 'TMEMO'
]


def load_huggingface_model(model_name, device):
    global hf_tokenizer, hf_model
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    hf_model.to(device)
    hf_model.eval()


def classify_document(file_path, device, metadata=None, mode="regex"):
    """
    Wrapper function to classify a document either via ML or regex.
    """
    if mode == "ml":
        try:
            return classify_with_ml(file_path, device, metadata)
        except Exception as e:
            print(
                f"[ML fallback] Classification failed: {e}. Falling back to regex.")
    return classify_with_regex(file_path)


def classify_with_regex(file_path):
    """
    Classify document using keyword matching in filename.
    """
    filename = file_path.name.lower()

    for doc_type, keywords in DOCUMENT_TYPES.items():
        for kw in keywords:
            if kw in filename:
                return doc_type
    return "REPORT"  # default fallback if nothing matches


def classify_with_ml(file_path, device, metadata=None):
    """
    Classify document using a fine-tuned Hugging Face transformer model.
    """
    if hf_tokenizer is None or hf_model is None:
        raise ValueError(
            "Hugging Face model not loaded. Call load_huggingface_model() first.")

    title = metadata.get("title", "").strip()

    inputs = hf_tokenizer(title, return_tensors="pt",
                          truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = hf_model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return class_names[predicted_class]
