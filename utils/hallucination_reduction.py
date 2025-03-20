# utils/hallucination_reduction.py
import re
import string
from collections import Counter

def detect_poor_ocr(text):
    """Detect if text likely contains OCR errors."""
    # Check character ratio (high special char rate indicates OCR issues)
    total_chars = len(text)
    if total_chars == 0:
        return True
        
    special_chars = sum(1 for c in text if c not in string.ascii_letters + string.digits + string.whitespace + string.punctuation)
    special_ratio = special_chars / total_chars
    
    # Check for common OCR patterns
    ocr_patterns = [
        r'[a-z][A-Z]{2,}[a-z]',  # Random capitalization
        r'\d[a-zA-Z]\d',         # Digit-letter-digit sequences
        r'[^\s\w\d.,;:?!-]'      # Unusual punctuation
    ]
    
    pattern_matches = sum(1 for pattern in ocr_patterns if re.search(pattern, text))
    
    return special_ratio > 0.05 or pattern_matches >= 2

def score_text_quality(text):
    """Score text quality from 0 (poor) to 1 (good)."""
    # Word repetition check
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0
        
    word_counts = Counter(words)
    repetition_ratio = max(word_counts.values()) / len(words) if words else 1
    
    # Sentence structure check - well-formed sentences end with punctuation
    sentences = re.split(r'[.!?]+', text)
    well_formed = sum(1 for s in sentences if len(s.strip()) > 10)
    structure_score = well_formed / len(sentences) if sentences else 0
    
    # OCR quality check
    ocr_penalty = 0.5 if detect_poor_ocr(text) else 0
    
    # Calculate final score (0-1)
    base_score = (1 - repetition_ratio) * 0.4 + structure_score * 0.6
    final_score = max(0, min(1, base_score - ocr_penalty))
    
    return final_score

def clean_ocr_text(text):
    """Clean common OCR artifacts from text."""
    # Remove repeated punctuation
    text = re.sub(r'([.!?,;:]){2,}', r'\1', text)
    
    # Fix I/l/1 confusion common in OCR
    text = re.sub(r'\bl\b', 'I', text)  # Single lowercase l is often 'I'
    
    # Remove garbage character sequences
    text = re.sub(r'[^\w\s.,;:?!-]{3,}', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def filter_low_quality_chunks(chunks, threshold=0.3):
    """Filter out low quality text chunks."""
    filtered_chunks = []
    for chunk in chunks:
        quality_score = score_text_quality(chunk)
        if quality_score >= threshold:
            # For medium quality text, try cleaning it
            if quality_score < 0.7:
                chunk = clean_ocr_text(chunk)
            filtered_chunks.append(chunk)
    
    return filtered_chunks if filtered_chunks else chunks  # Fall back to original if all filtered out
