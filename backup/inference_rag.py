import os
import argparse
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import re
import datetime
import json
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.query_utils import detect_file_references, build_metadata_filter
from utils.context_analysis import rerank_documents
def clean_response(response, query):
    """Remove irrelevant content after the main answer"""
    # If response contains numbered points, keep only those sections
    if re.search(r'\d+\.', response):
        # Extract the numbered sections plus one concluding paragraph
        main_content = re.split(r'\n\n(?![\d\.])', response, maxsplit=1)[0]
        return main_content
        
    # For responses without numbered points, keep only the first 2-3 paragraphs
    paragraphs = response.split('\n\n')
    if len(paragraphs) > 3:
        return '\n\n'.join(paragraphs[:3])
    
    return response



def log_query_results(query, context, sources, response, log_file="query_log.jsonl"):
    """Log query results for later analysis"""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "source_files": [src.get("filename", "unknown") for src in sources],
        "response": response
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    print(f"Query logged to {log_file}")


def post_process_response(response):
    """Clean up the response to remove artifacts and improve formatting."""
    # Remove special characters often used as section breaks
    response = re.sub(r'[╩§●■►▼▲◄]', '', response)
    
    # Remove trailing section headers (starts with ###)
    response = re.sub(r'\n+### [^\n]+$', '', response)
    
    # If response contains points followed by unrelated content, try to trim it
    if re.search(r'[d]\)\s*[^\n]+\n\n', response):
        # Try to trim after the last main point (assuming format a), b), c), d))
        match = re.search(r'([d]\)\s*[^\n]+)\n\n', response)
        if match:
            end_pos = match.end(1)
            response = response[:end_pos] + "\n\nIn conclusion, these are the four main purposes for adopting the OH&S Standard."
    
    return response.strip()



def load_model(model_path, device="auto", use_lora=True, base_model=None):
    """Load the fine-tuned model."""
    # Determine if we're loading a LoRA model or a full model
    if use_lora:
        if base_model is None:
            # Try to get the base model from the LoRA config
            config = PeftConfig.from_pretrained(model_path)
            base_model = config.base_model_name_or_path
        
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
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
    """Load the vector store from the given path."""
    print(f"Loading vector store from {vector_store_path}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store
'''
def retrieve_context(vector_store, query, k=3):
    """Retrieve the most relevant documents for the query."""
    print(f"Retrieving context for query: {query}")
    retrieved_docs = vector_store.similarity_search(query, k=k)
    
    print("Retrieved Docs:", retrieved_docs)

    # Clean up the context to remove special tags
    clean_context = []
    for doc in retrieved_docs:
        content = doc.page_content
        # Remove special tags that might confuse the model
        content = re.sub(r'<<[^>]+>>', '', content)
        content = re.sub(r'<<[^>]+_END>>', '', content)
        clean_context.append(content)
    

    context = "\n\n".join(clean_context)
    sources = [doc.metadata for doc in retrieved_docs]
    
    print(f"Retrieved Context: {context}")


    return context, sources
'''
def select_relevant_chunks(context, query):
    """Extract only highly relevant content from the context"""
    paragraphs = context.split('\n\n')
    
    # Score paragraphs by relevance to query
    relevant_paragraphs = []
    query_terms = query.lower().split()
    
    for para in paragraphs:
        # Simple relevance scoring
        relevance = sum(1 for term in query_terms if term in para.lower())
        if relevance > 0:
            relevant_paragraphs.append(para)
    
    # If we found relevant paragraphs, use only those
    if relevant_paragraphs:
        return '\n\n'.join(relevant_paragraphs[:5])  # Limit to top 5
    
    # Otherwise use first few paragraphs
    return '\n\n'.join(paragraphs[:3])


# Replace your retrieve_context function with this enhanced version
def retrieve_context(vector_store, query, k=3):
    """Retrieve the most relevant documents for the query with metadata filtering."""
    print(f"Retrieving context for query: {query}")
    

    # Retrieve more documents than we need for reranking
    k_retrieval = min(k * 2, 10)  # Retrieve up to 2x documents for reranking
    

    # Detect potential file references
    file_references = detect_file_references(query)
    metadata_filter = None
    
    if file_references:
        print(f"Detected file references: {file_references}")
        metadata_filter = build_metadata_filter(file_references)
    
    # Try metadata-filtered search first if we have file references
    retrieved_docs = None
    if metadata_filter:
        try:
            retrieved_docs = vector_store.similarity_search(
                query,
                k=k_retrieval,
                filter=metadata_filter
            )
            if retrieved_docs:
                print(f"Retrieved {len(retrieved_docs)} docs using metadata filtering")
        except Exception as e:
            print(f"Error with metadata filtering: {e}")
            retrieved_docs = None
    
    # Fall back to regular search if needed
    if not retrieved_docs:
        retrieved_docs = vector_store.similarity_search(query, k=k_retrieval)

    # Rerank documents if we retrieved more than we need
    if len(retrieved_docs) > k:
        print(f"Reranking {len(retrieved_docs)} documents...")
        retrieved_docs = rerank_documents(retrieved_docs, query, top_k=k)


    
    # Clean up the context to remove special tags
    clean_context = []
    for doc in retrieved_docs:
        content = doc.page_content
        # Remove special tags that might confuse the model
        content = re.sub(r'<<[^>]+>>', '', content)
        content = re.sub(r'<<[^>]+_END>>', '', content)
        clean_context.append(content)
    
    # Use the clean context (this was the issue we fixed earlier)
    context = "\n\n".join(clean_context)
    sources = [doc.metadata for doc in retrieved_docs]
    
    print(f"Retrieved Context: {context}")
    
    return context, sources


def generate_rag_response(model, tokenizer, query, context, max_length=2048, temperature=0.7, top_p=0.9):
    """Generate a response using the retrieved context and the query."""
    # Format the prompt with the context and query
    formatted_prompt = f"<s>[INST] I need information about the following topic. Use the provided context to answer my question accurately and concisely.  \n\nWhen listing specific points, format them clearly with numbers or letters as shown in the context, and provide a brief conclusion or summary at the end of your response.\n\n Provide ONLY information directly relevant to my question. \n\nContext:\n{context}\n\nQuestion: {query} [/INST]"
    

    print(f"DEBUG: Prompt length: {len(formatted_prompt)}")


    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate a response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the model's response (after [/INST])
#    response = response.split("[/INST]")[-1].strip()
    
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    else:
        print("WARNING: [/INST] tag not found in response")
        response = ""

    # Remove the final </s> if present
    if response.endswith("</s>"):
        response = response[:-4].strip()
    

    # Clean up any custom tags
    response = re.sub(r'\[ANS\]|\[/AN.*?$', '', response).strip()


    # Fallback for empty responses
    if len(response) < 20:
        print("WARNING: Empty or very short response, using fallback")
        response = extract_key_points(context, query)

    return response


def extract_key_points(context, query):
    """Extract key points from context when model fails to generate response."""
    # Split context into sentences
    sentences = re.split(r'(?<=[.!?])\s+', context)
    
    # Look for sentences containing keywords from the query

    keywords = [word.lower() for word in query.lower().split() if len(word) > 3]
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        matches = sum(1 for keyword in keywords if keyword in sentence_lower)
        if matches > 0 and len(sentence) > 20:
            relevant_sentences.append((sentence, matches))


    relevant_sentences.sort(key=lambda x: x[1], reverse=True)

    selected_sentences = [s[0] for s in relevant_sentences[:3]]



    if selected_sentences:
        response = "Based on the provided documents, I found the following information:\n\n"
        for i, sentence in enumerate(selected_sentences):
            response += f"{i+1}. {sentence}\n"
    # Otherwise use the first few sentences
    else:
        response = "The documents contain the following information:\n\n"
        for i, sentence in enumerate(sentences[:3]):
            if len(sentence) > 20:
                response += f"{i+1}. {sentence}\n"
    
    return response




def run_interactive_rag_console(model, tokenizer, vector_store, k_docs=3, verbose=False):
    """Run an interactive console for RAG-based chatting with the model."""
    print("Interactive RAG console starting. Type 'exit' to quit.")
    print("-----------------------------------------------------")
    
    while True:
        # Get user input
        user_input = input("\nUser: ")
        
        # Check if the user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Retrieve context
        context, sources = retrieve_context(vector_store, user_input, k=k_docs)
        if verbose:
            print("\nRetrieved Context:", context)


        # Generate a response
        print("\nGenerating response...")
        response = generate_rag_response(model, tokenizer, user_input, context)
        

        # Debugging: Check what response is generated

        if verbose:
            print("\nDEBUG: Model Output:", response)


        # Print the response and sources
        print(f"\nModel: {response}")
        print("\nSources:")
        for i, source in enumerate(sources):
            print(f"  {i+1}. {source.get('title', 'Untitled')} (from {source.get('filename', 'unknown')})")
        # Log the query and response
        log_query_results(user_input, context, sources, response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG inference with the fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--vector_store_path", type=str, required=True, help="Path to the vector store")
    parser.add_argument("--device", type=str, default="auto", help="Device to run inference on")
    parser.add_argument("--use_lora", action="store_true", help="Whether the model is a LoRA model")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name if using LoRA")
    parser.add_argument("--k_docs", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug output")
    parser.add_argument("--log_file", type=str, default="query_log.jsonl", help="File to log queries and responses")



    args = parser.parse_args()
    
    # Load the model
    model, tokenizer = load_model(args.model_path, args.device, args.use_lora, args.base_model)
    
    # Load the vector store
    vector_store = load_vector_store(args.vector_store_path)
    
    # Run the interactive console
    run_interactive_rag_console(model, tokenizer, vector_store, k_docs=args.k_docs, verbose=args.verbose)
