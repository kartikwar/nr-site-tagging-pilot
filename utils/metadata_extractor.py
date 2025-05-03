def extract_metadata(file_path):
    """Return dummy metadata."""
    metadata = {
        "site_id": "12345",
        "address": "N/A",
        "sender": "Unknown",
        "receiver": "Unknown"
    }
    print(f"[Metadata] Extracted from {file_path.name}: {metadata}")
    return metadata