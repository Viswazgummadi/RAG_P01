import os
import argparse
import json
from tqdm import tqdm
import sys

# Add project root to path so we can import from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import preprocess_tagged_text
from vector_db import build_vector_store

def find_all_text_files(root_dir):
    """Find all .txt files recursively in the given directory."""
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    return all_files

def process_all_files(input_dir, output_jsonl, vector_db_dir, chunk_size=1000, chunk_overlap=200):
    """Process all text files and build a vector database."""
    # Find all text files
    print(f"Finding all text files in {input_dir}...")
    text_files = find_all_text_files(input_dir)
    print(f"Found {len(text_files)} text files.")
    
    # Process each file
    all_documents = []
    error_files = []
    
    for file_path in tqdm(text_files, desc="Processing files"):
        try:
            print(f"Processing: {file_path}")
            doc_data = preprocess_tagged_text(file_path)
            all_documents.append(doc_data)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            error_files.append((file_path, str(e)))
    
    # Save as JSONL
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for doc in all_documents:
            f.write(json.dumps(doc) + '\n')
    
    print(f"Processed {len(all_documents)} documents and saved to {output_jsonl}")
    
    # Log errors if any
    if error_files:
        error_log = os.path.join(os.path.dirname(output_jsonl), "processing_errors.txt")
        with open(error_log, 'w', encoding='utf-8') as f:
            for file_path, error in error_files:
                f.write(f"{file_path}\t{error}\n")
        print(f"Encountered errors in {len(error_files)} files. See {error_log} for details.")
    
    # Build vector database
    print("Building vector database...")
    build_vector_store(output_jsonl, vector_db_dir, chunk_size, chunk_overlap)
    
    return all_documents

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process all text files and build vector database")
    parser.add_argument("--input_dir", type=str, default="data2", help="Root directory containing text files")
    parser.add_argument("--output_jsonl", type=str, default="processed_data/all_documents.jsonl", help="Output JSONL file")
    parser.add_argument("--vector_db_dir", type=str, default="vector_db", help="Vector database output directory")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of text chunks")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap between chunks")
    
    args = parser.parse_args()
    process_all_files(args.input_dir, args.output_jsonl, args.vector_db_dir, args.chunk_size, args.chunk_overlap)
