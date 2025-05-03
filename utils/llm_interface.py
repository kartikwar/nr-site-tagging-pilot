import ollama
import re

def query_llm(prompt, model="llama2", system_prompt=None):
    """Query LLaMA via Ollama and return a cleaned site ID."""
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    try:
        response = ollama.chat(model=model, messages=messages)
        raw = response['message']['content'].strip()
    except Exception as e:
        return "UNKNOWN"

    # Extract the first standalone number from the response
    match = re.search(r"\b\d{3,5}\b", raw)
    if match:
        return match.group(0)
    else:
        return "UNKNOWN"