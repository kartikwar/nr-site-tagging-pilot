import pandas as pd

def load_gold_data(file_path, csv_path):
    """Load gold metadata for training examples."""
    df = pd.read_csv(csv_path, encoding='windows-1252', header=3)
    match = df[df['Current BC Mail title'] == file_path]

    if match.empty:
        return f"No matching entry for '{file_path}' in gold data."
    
    row = match.iloc[0]

    metadata = {
        'title': str(row.get('Title/Subject', '')),
        'receiver': str(row.get('Receiver', '')),
        'sender': str(row.get('Sender/Author', '')),
        'address': str(row.get('Address ', '')),
        'site_id': str(row.get('Site ID', ''))
    }

    return metadata