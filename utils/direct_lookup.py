# utils/direct_lookup.py
import os
import re

def find_file_by_standard_number(standard_number, data_dir="data2"):
    """Find a file containing a specific standard number."""
    standard_pattern = re.sub(r'[^0-9]', '', standard_number)  # Extract numbers
    matching_files = []
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.txt') and standard_pattern in file:
                matching_files.append(os.path.join(root, file))
    
    return matching_files

def extract_standard_from_query(query):
    """Extract standard number from a query."""
    # Match ISO/IEC/BS standard patterns
    patterns = [
        r'(?:ISO|IEC|BS)[- ]?(\d+(?:-\d+)?):?(\d{4})?',
        r'(?:ISO|IEC|BS)[- ]?(\d+)[ -](\d+):?(\d{4})?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(0)
    
    return None
