import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_vector_store(vector_store_path):
    """Load the vector store from the given path."""
    print(f"Loading vector store from {vector_store_path}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    vector_store = FAISS.load_local(vector_store_path, embeddings,allow_dangerous_deserialization=True)
    return vector_store

def retrieve_context(vector_store, query, k=3):
    """Retrieve the most relevant documents for the query."""
    retrieved_docs = vector_store.similarity_search(query, k=k)
    
    # Clean up the context
    clean_context = []
    for doc in retrieved_docs:
        content = doc.page_content
        # Remove special tags
        content = re.sub(r'<<[^>]+>>', '', content)
        content = re.sub(r'<<[^>]+_END>>', '', content)
        clean_context.append(content)
    
    context = "\n\n".join(clean_context)
    sources = [doc.metadata for doc in retrieved_docs]
    
    return context, sources

def main(args):
    # Load model and tokenizer
    print(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load vector store
    vector_store = load_vector_store(args.vector_store_path)
    
    while True:
        # Get user input
        user_input = input("\nUser: ")
        
        # Check if the user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Retrieve context
        context, sources = retrieve_context(vector_store, user_input, k=args.k_docs)
        
        # Format prompt
        prompt = f"<s>[INST] I need information about the following topic. Use the provided context to answer my question accurately.\n\nContext:\n{context}\n\nQuestion: {user_input} [/INST]"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True
            )
        
        # Process response
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        if response.endswith("</s>"):
            response = response[:-4].strip()
        
        print(f"\nModel: {response}")
        print("\nSources:")
        for i, source in enumerate(sources):
            print(f"  {i+1}. {source.get('title', 'Untitled')} (from {source.get('filename', 'unknown')})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test base model with RAG")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Base model to use")
    parser.add_argument("--vector_store_path", type=str, required=True, help="Path to the vector store")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for generation")
    parser.add_argument("--k_docs", type=int, default=2, help="Number of documents to retrieve")
    
    args = parser.parse_args()
    main(args)
