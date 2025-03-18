import os
import re
import json
from tqdm import tqdm

def preprocess_tagged_text(text_file_path):
    """Process text files with special tags like <<DOCUMENT_START>>, <<HEADING level=X>>, etc."""
    with open(text_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract document metadata
    metadata = {}
    metadata['filename'] = os.path.basename(text_file_path)
    metadata['source_path'] = text_file_path
    
    # Extract metadata section
    metadata_match = re.search(r'<<SECTION_START:METADATA>>(.*?)<<SECTION_END>>', content, re.DOTALL)
    if metadata_match:
        metadata_text = metadata_match.group(1)
        for line in metadata_text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
    
    # Remove document markers
    content = re.sub(r'<<DOCUMENT_START>>', '', content)
    content = re.sub(r'<<DOCUMENT_END>>', '', content)
    
    # Extract title
    title_match = re.search(r'<<HEADING level=[12]>>(.*?)<<HEADING_END>>', content)
    if title_match:
        metadata['title'] = title_match.group(1).strip()
    else:
        metadata['title'] = "Untitled Document"
    
    # Process page markers
    content = re.sub(r'<<PAGE_\d+>>', '', content)
    content = re.sub(r'<<PAGE_END>>', '', content)
    
    # Convert headings to markdown
    def heading_replacer(match):
        level = match.group(1)
        text = match.group(2)
        markdown_heading = '#' * int(level) + ' ' + text.strip()
        return markdown_heading
    
    content = re.sub(r'<<HEADING level=(\d+)>>(.*?)<<HEADING_END>>', heading_replacer, content)
    
    # Process paragraphs
    content = re.sub(r'<<PARAGRAPH>>', '', content)
    content = re.sub(r'<<PARAGRAPH_END>>', '\n\n', content)
    
    # Remove section markers
    content = re.sub(r'<<SECTION_START:[^>]+>>', '', content)
    content = re.sub(r'<<SECTION_END>>', '', content)
    
    # Clean up extra whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.strip()
    
    return {
        'content': content,
        'metadata': metadata
    }

def process_directory(input_dir, output_file):
    """Process all text files in a directory and save to structured JSONL format."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    all_documents = []
    
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc="Processing files"):
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    doc_data = preprocess_tagged_text(file_path)
                    all_documents.append(doc_data)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    # Save as JSONL for easier processing
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in all_documents:
            f.write(json.dumps(doc) + '\n')
    
    print(f"Processed {len(all_documents)} documents and saved to {output_file}")
    return all_documents

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process tagged text files into structured format")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing text files")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file path")
    
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_file)
