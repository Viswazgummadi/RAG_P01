from langchain_community.vectorstores import FAISS

#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json

import os
import torch
from tqdm import tqdm

def build_vector_store(jsonl_file, output_dir, chunk_size=1000, chunk_overlap=200):
    """Build a vector store from processed JSONL data."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize embedding model
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
    )
    
    documents = []
    
    # Load and process documents
    print("Loading documents and splitting into chunks...")
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            doc_data = json.loads(line)
            content = doc_data['content']
            metadata = doc_data['metadata']
            
            # Split content into chunks
            chunks = text_splitter.split_text(content)
            
            # Create Document objects with metadata
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'title': metadata.get('title', 'Untitled'),
                        'filename': metadata.get('filename', ''),
                        'author': metadata.get('author', ''),
                        'date': metadata.get('date', ''),
                        'chunk_id': i,
                        'source_path': metadata.get('source_path', '')
                    }
                )
                documents.append(doc)
    
    print(f"Created {len(documents)} document chunks for vectorization")
    
    # Create vector store
    print("Building vector store. This may take a while...")
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save vector store
    print(f"Saving vector store to {output_dir}")
    vector_store.save_local(output_dir)
    print(f"Vector store saved successfully with {vector_store.index.ntotal} vectors")
    
    return vector_store

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build vector database from processed documents")
    parser.add_argument("--jsonl_file", type=str, required=True, help="Path to processed JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save vector store")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of text chunks")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap between chunks")
    
    args = parser.parse_args()
    build_vector_store(args.jsonl_file, args.output_dir, args.chunk_size, args.chunk_overlap)
