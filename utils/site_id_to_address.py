import re
import pandas as pd
import string
from rapidfuzz import fuzz

def clean_address(addr):
    if not isinstance(addr, str):
        return ''
    addr = addr.lower().strip()
    addr = addr.translate(str.maketrans('', '', string.punctuation))
    return addr

def extract_numbers(s):
    return set(re.findall(r'\d+', s))

def format_address(row, threshold=85):
    addr1_raw = row['Address 1']
    addr2_raw = row['Address 2']
    urban_area = row['Urban Area']
    # Drop postal code if not present
    if str(row['Postal Code']) == 'No Entry' or str(row['Postal Code']) == 'nan':
        postal_code = None
    else:
        postal_code = row['Postal Code']
    
    addr1 = clean_address(addr1_raw)
    addr2 = clean_address(addr2_raw)

    # Case 1: 'No Entry' in Address 2 - drop it
    if not addr2 or 'no entry' in addr2:
        if postal_code:
            return f"{addr1_raw}, {urban_area}, {postal_code}"
        else:
            return f"{addr1_raw}, {urban_area}"
    
    # Case 2: Address 2 contains numerical information not present in Address 1 (eg address numbers) - keep it
    nums1 = extract_numbers(addr1)
    nums2 = extract_numbers(addr2)
    if nums2 - nums1:  # If Address 2 has a number not in Address 1
        if postal_code:
            return f"{addr1_raw}, {addr2_raw}, {urban_area}, {postal_code}"
        else:
            return f"{addr1_raw}, {addr2_raw}, {urban_area}"
        
    
    # Case 3: Redundant addresses based on fuzzy match - drop least informative (shortest string) Address
    score = fuzz.token_set_ratio(addr1, addr2)
    if score >= threshold:
        preferred = addr1_raw if len(addr1_raw) >= len(addr2_raw) else addr2_raw
        if postal_code:
            return f"{preferred}, {urban_area}, {postal_code}"
        else:
            return f"{preferred}, {urban_area}"
    
    # Case 3: Distinct addresses
    if postal_code:
        return f"{addr1_raw}, {addr2_raw}, {urban_area}, {postal_code}"
    else:
        return f"{addr1_raw}, {addr2_raw}, {urban_area}"

# Load CSV and apply to a specific Site ID
def get_site_address(csv_path, site_id):
    df = pd.read_csv(csv_path)
    row = df[df['Site ID'] == site_id]
    return format_address(row.iloc[0])
