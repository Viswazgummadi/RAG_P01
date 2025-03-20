# utils/document_structure.py
import re

def identify_document_sections(text):
    """Identify common sections in documents."""
    section_patterns = {
        'scope': [
            r'(?i)(?:\d+\s+)?scope\s*\n',
            r'(?i)(?:\d+\s+)?scope\s+and\s+field\s+of\s+application',
            r'(?i)1\s+scope'
        ],
        'normative_references': [
            r'(?i)(?:\d+\s+)?normative\s+references',
            r'(?i)(?:\d+\s+)?references'
        ],
        'terms_and_definitions': [
            r'(?i)(?:\d+\s+)?terms\s+and\s+definitions',
            r'(?i)(?:\d+\s+)?definitions'
        ],
        'requirements': [
            r'(?i)(?:\d+\s+)?requirements',
            r'(?i)(?:\d+\s+)?general\s+requirements'
        ],
        'introduction': [
            r'(?i)(?:\d+\s+)?introduction',
            r'(?i)(?:\d+\s+)?overview'
        ],
        'summary': [
            r'(?i)(?:\d+\s+)?summary',
            r'(?i)(?:\d+\s+)?executive\s+summary'
        ],
        'conclusion': [
            r'(?i)(?:\d+\s+)?conclusion[s]?',
            r'(?i)(?:\d+\s+)?findings'
        ]
    }
    
    sections = {}
    for section_name, patterns in section_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                # Find the start of the section
                start_pos = match.start()
                
                # Look for the next section header to determine the end
                next_section_pattern = r'(?i)(?:\d+\s+)?[a-z\s]+\s*\n'
                next_match = re.search(next_section_pattern, text[start_pos + len(match.group()):])
                if next_match:
                    end_pos = start_pos + len(match.group()) + next_match.start()
                    sections[section_name] = text[start_pos:end_pos].strip()
                else:
                    # If no next section, take a reasonable chunk
                    end_pos = min(start_pos + 2000, len(text))
                    sections[section_name] = text[start_pos:end_pos].strip()
                break
    
    return sections

def extract_section_from_text(text, section_name):
    """Extract a specific section from document text."""
    sections = identify_document_sections(text)
    return sections.get(section_name, None)

def detect_section_request(query):
    """Detect if query is asking for a specific document section."""
    section_keywords = {
        'scope': ['scope', 'purpose', 'objective', 'field of application'],
        'normative_references': ['normative references', 'references', 'standards referenced'],
        'terms_and_definitions': ['terms', 'definitions', 'terminology'],
        'requirements': ['requirements', 'specifications', 'criteria'],
        'introduction': ['introduction', 'overview', 'background'],
        'summary': ['summary', 'executive summary', 'abstract'],
        'conclusion': ['conclusion', 'findings', 'final remarks']
    }
    
    query_lower = query.lower()
    for section, keywords in section_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                return section
    
    return None
