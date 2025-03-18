import os
import sys
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def test_vector_db(vector_db_path):
    """Test the vector database with a few queries."""
    print(f"Loading vector store from {vector_db_path}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    vector_store = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    
    # Get total number of vectors
    print(f"Vector store loaded. Total vectors: {vector_store.index.ntotal}")
    
    # Test queries
    test_queries = [
        "What are the requirements for cocoa bean storage?",
        "How should documents be formatted?",
        "What are the key components of the system?",
        "Tell me about storage structures"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        docs = vector_store.similarity_search(query, k=2)
        
        print(f"Found {len(docs)} documents:")
        for i, doc in enumerate(docs):
            print(f"Document {i+1}:")
            print(f"  Source: {doc.metadata.get('source_path', 'unknown')}")
            print(f"  Title: {doc.metadata.get('title', 'Untitled')}")
            print(f"  Content preview: {doc.page_content[:150]}...")
            print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test vector database")
    parser.add_argument("--vector_db_path", type=str, default="vector_db", help="Path to vector database")
    
    args = parser.parse_args()
    test_vector_db(args.vector_db_path)
