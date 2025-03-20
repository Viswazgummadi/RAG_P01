# scripts/test_rag_quality.py
import os
import sys
import json
import argparse
import torch
import re
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils.query_utils import detect_file_references, build_metadata_filter
from utils.context_analysis import rerank_documents

def load_model(model_path, device="auto", use_lora=True, base_model=None):
    """Load the model."""
    # Similar to your existing load_model function
    if use_lora:
        if base_model is None:
            # Try to get the base model from the LoRA config
            try:
                config = PeftConfig.from_pretrained(model_path)
                base_model = config.base_model_name_or_path
            except Exception as e:
                print(f"Error loading PEFT config: {e}")
                if not base_model:
                    raise ValueError("Base model must be provided if adapter config not found")
        
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print(f"Loading LoRA adapter: {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
        else:
            print(f"Warning: No adapter config found in {model_path}, using base model only")
    else:
        print(f"Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path if not use_lora else base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_vector_store(vector_store_path):
    """Load the vector store."""
    print(f"Loading vector store from {vector_store_path}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

def retrieve_context(vector_store, query, k=3):
    """Retrieve context with all the enhancements."""
    # Implementation similar to the one in inference_rag.py with reranking
    # Detect file references
    file_references = detect_file_references(query)
    metadata_filter = None
    
    if file_references:
        print(f"Detected file references: {file_references}")
        metadata_filter = build_metadata_filter(file_references)
    
    # Try metadata-filtered search first
    retrieved_docs = None
    if metadata_filter:
        try:
            retrieved_docs = vector_store.similarity_search(
                query,
                k=k*2,
                filter=metadata_filter
            )
        except Exception as e:
            print(f"Error with metadata filtering: {e}")
    
    # Fall back to regular search if needed
    if not retrieved_docs:
        retrieved_docs = vector_store.similarity_search(query, k=k*2)
    
    # Rerank documents
    if len(retrieved_docs) > k:
        retrieved_docs = rerank_documents(retrieved_docs, query, top_k=k)
    
    # Clean up the context
    clean_context = []
    for doc in retrieved_docs:
        content = doc.page_content
        content = re.sub(r'<<[^>]+>>', '', content)
        content = re.sub(r'<<[^>]+_END>>', '', content)
        clean_context.append(content)
    
    context = "\n\n".join(clean_context)
    sources = [doc.metadata for doc in retrieved_docs]
    
    return context, sources, retrieved_docs

def generate_response(model, tokenizer, query, context, max_length=2048, temperature=0.7):
    """Generate response from model."""
    # Implementation similar to generate_rag_response in inference_rag.py
    formatted_prompt = f"<s>[INST] I need information about the following topic. Use the provided context to answer my question accurately.\n\nContext:\n{context}\n\nQuestion: {query} [/INST]"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    
    if response.endswith("</s>"):
        response = response[:-4].strip()
    
    return response

def evaluate_file_retrieval(retrieved_docs, expected_files):
    """Evaluate if the retrieved documents match the expected files."""
    if not expected_files:
        return True, 1.0
    
    retrieved_files = [doc.metadata.get('filename', '').lower() for doc in retrieved_docs]
    
    # Check if any expected file pattern is in any retrieved filename
    matches = 0
    for expected in expected_files:
        expected_lower = expected.lower()
        for filename in retrieved_files:
            if expected_lower in filename:
                matches += 1
                break
    
    accuracy = matches / len(expected_files) if expected_files else 1.0
    success = accuracy > 0  # At least one match
    
    return success, accuracy

def main(args):
    """Run evaluation on test questions."""
    # Load model
    model, tokenizer = load_model(
        args.model_path, 
        use_lora=args.use_lora, 
        base_model=args.base_model
    )
    
    # Load vector store
    vector_store = load_vector_store(args.vector_store_path)
    
    # Load test questions
    with open(args.test_file, 'r') as f:
        test_questions = json.load(f)
    
    # Results container
    results = []
    
    # Process each question
    for question in tqdm(test_questions, desc="Evaluating questions"):
        query = question["question"]
        expected_files = question.get("expected_files", [])
        
        # Retrieve context
        context, sources, retrieved_docs = retrieve_context(
            vector_store, 
            query, 
            k=args.k_docs
        )
        
        # Generate response
        response = generate_response(
            model,
            tokenizer,
            query,
            context,
            temperature=args.temperature
        )
        
        # Evaluate file retrieval
        file_match_success, file_match_accuracy = evaluate_file_retrieval(
            retrieved_docs, 
            expected_files
        )
        
        # Record result
        result = {
            "question": query,
            "expected_files": expected_files,
            "retrieved_files": [doc.metadata.get('filename', '') for doc in retrieved_docs],
            "file_match_success": file_match_success,
            "file_match_accuracy": file_match_accuracy,
            "response": response,
            "sources": [
                {
                    "title": src.get("title", "Untitled"),
                    "filename": src.get("filename", "unknown")
                }
                for src in sources
            ]
        }
        
        results.append(result)
    
    # Calculate overall metrics
    successful_retrievals = sum(1 for r in results if r["file_match_success"])
    retrieval_accuracy = successful_retrievals / len(results)
    
    # Add summary to results
    summary = {
        "total_questions": len(results),
        "successful_retrievals": successful_retrievals,
        "retrieval_accuracy": retrieval_accuracy
    }
    
    output = {
        "summary": summary,
        "results": results
    }
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"Total questions: {len(results)}")
    print(f"Successful retrievals: {successful_retrievals}/{len(results)} ({retrieval_accuracy:.2%})")
    
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG system quality")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model or model name")
    parser.add_argument("--vector_store_path", type=str, required=True,
                        help="Path to the vector store")
    parser.add_argument("--test_file", type=str, required=True,
                        help="JSON file with test questions")
    parser.add_argument("--output_file", type=str, required=True,
                        help="JSON file to save results")
    parser.add_argument("--use_lora", action="store_true",
                        help="Whether the model is a LoRA model")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model name if using LoRA")
    parser.add_argument("--k_docs", type=int, default=3,
                        help="Number of documents to retrieve")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    
    args = parser.parse_args()
    main(args)
