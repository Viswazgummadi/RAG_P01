# utils/query_utils.py
import re

def detect_file_references(query):
    """Extract potential file references from the query."""
    # Simple regex pattern to detect file mentions
    file_patterns = [
        r'file[s]?\s+(?:called|named)?\s+[\'"]?([^\'".,\s]+)[\'"]?',
        r'document[s]?\s+(?:called|named)?\s+[\'"]?([^\'".,\s]+)[\'"]?',
        r'in\s+[\'"]?([^\'".,\s]+\.(txt|pdf|doc|docx))[\'"]?',
        r'from\s+[\'"]?([^\'".,\s]+)[\'"]?',
        r'([^\'".,\s]+\.(txt|pdf|doc|docx))'  # Detect raw filenames
    ]
    
    potential_files = []
    for pattern in file_patterns:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            potential_files.append(match.group(1))
            
    # Remove duplicates while preserving order
    seen = set()
    return [x for x in potential_files if not (x in seen or seen.add(x))]


def get_available_files(vector_store):
    """Get a list of all unique filenames in the vector store metadata."""
    available_files = set()
    for doc in vector_store.docstore._dict.values():
        if hasattr(doc, 'metadata') and 'filename' in doc.metadata:
            available_files.add(doc.metadata['filename'])
    return list(available_files)


def find_best_matching_file(file_reference, available_files):
    """Find the best matching filename from available files."""
    # Exact match
    for file in available_files:
        if file.lower() == file_reference.lower():
            return file
    
    # Partial match
    best_matches = []
    for file in available_files:
        if file_reference.lower() in file.lower():
            best_matches.append(file)
    
    if best_matches:
        # Return the shortest matching filename (likely most relevant)
        return min(best_matches, key=len)
    
    return None




def build_metadata_filter(file_references=None, available_files=None):
    """Build a metadata filter dict for vector search."""
    if not file_references or not available_files:
        return None
    
    matched_files = []
    for ref in file_references:
        match = find_best_matching_file(ref, available_files)
        if match:
            matched_files.append(match)
    
    if not matched_files:
        return None
    
    # Create filter dict
    file_filters = []
    for file in matched_files:
        file_filters.append({"filename": file})
    
    if len(file_filters) == 1:
        return file_filters[0]
    else:
        return {"$or": file_filters}






def detect_section_references(query):
    """Extract potential section references from the query."""
    section_patterns = [
        r'section[s]?\s+(?:called|named|titled)?\s+[\'"]?([^\'".,\s]+)[\'"]?',
        r'(?:in|from) the\s+([a-z]+tion)\s+(?:section|part)',
        r'(?:in|from) the\s+([a-z]+ter)\s+(?:section|part)',
    ]
    
    potential_sections = []
    for pattern in section_patterns:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            potential_sections.append(match.group(1))
            
    return potential_sections


'''
def build_metadata_filter(file_references=None, section_references=None):
    """Build a metadata filter for vector search."""
    metadata_filter = {}
    
    # Add file filters
    if file_references and len(file_references) > 0:
        file_filters = []
        for file_ref in file_references:
            file_filters.append({"filename": {"$regex": file_ref}})
        
        if len(file_filters) > 1:
            metadata_filter["$or"] = file_filters
        else:
            metadata_filter = file_filters[0]
    
    # Add section filters if your metadata includes section info
    # This would need to be customized based on your metadata structure
    
    return metadata_filter
'''
