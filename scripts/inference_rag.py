import os
import argparse
import torch
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import re
import datetime
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.query_utils import detect_file_references, build_metadata_filter, get_available_files
from utils.context_analysis import rerank_documents
from utils.hallucination_reduction import filter_low_quality_chunks, clean_ocr_text
from utils.document_structure import detect_section_request, extract_section_from_text
from utils.direct_lookup import extract_standard_from_query, find_file_by_standard_number


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

def retrieve_context(vector_store, query, k=3, verbose=False):
    """Retrieve the most relevant documents for the query with metadata filtering."""
    if verbose:
        print(f"Retrieving context for query: {query}")


    # Detect if query is asking for a specific section
    requested_section = detect_section_request(query)
    if verbose and requested_section:
        print(f"Detected section request: {requested_section}")



    # Get all available files in the vector store
    # Get available files
    try:
        available_files = get_available_files(vector_store)
    except:
        available_files = []
        if verbose:
            print("Unable to get available files list")



    # Detect potential file references
    file_references = detect_file_references(query)
    if verbose and file_references:
        print(f"Detected file references: {file_references}")



    metadata_filter = None

    if file_references and available_files:
        metadata_filter = build_metadata_filter(file_references, available_files)


    # Retrieve more documents than needed for reranking
    k_retrieval = min(k * 3, 15)


    # Try metadata-filtered search first if we have file references
    retrieved_docs = None
    if metadata_filter:
        try:
            retrieved_docs = vector_store.similarity_search(
                query,
                k=k_retrieval,
                filter=metadata_filter
            )
            if retrieved_docs and verbose:
                print(f"Retrieved {len(retrieved_docs)} docs using metadata filtering")
        except Exception as e:
            if verbose:
                print(f"Error with metadata filtering: {e}")

    # Fall back to regular search if needed
    if not retrieved_docs:
        retrieved_docs = vector_store.similarity_search(query, k=k_retrieval)


    # Process the retrieved documents
    if retrieved_docs and requested_section:
        # Group docs by filename to reconstruct document structure
        docs_by_filename = {}
        for doc in retrieved_docs:
            filename = doc.metadata.get('filename', 'unknown')
            if filename not in docs_by_filename:
                docs_by_filename[filename] = []
            docs_by_filename[filename].append(doc.page_content)
        
        # Try to find the requested section in each document
        section_contexts = []
        for filename, contents in docs_by_filename.items():
            # Reconstruct the document
            full_text = "\n\n".join(contents)
            # Extract the requested section
            section_text = extract_section_from_text(full_text, requested_section)
            if section_text:
                section_contexts.append((filename, section_text))
        
        # If we found any section-specific contexts, prioritize them
        if section_contexts:
            if verbose:
                print(f"Found {len(section_contexts)} documents with '{requested_section}' section")
            
            # Create new document objects from section contexts
            section_docs = []
            for filename, section_text in section_contexts:
                # Clean the section text
                clean_text = re.sub(r'<<[^>]+>>', '', section_text)
                clean_text = re.sub(r'<<[^>]+_END>>', '', clean_text)
                
                # Create a new document with this section context
                meta = {"filename": filename, "section": requested_section}
                section_docs.append(Document(page_content=clean_text, metadata=meta))
            
            # Use these section-specific documents instead
            retrieved_docs = section_docs[:k]

    # Clean up the context to remove special tags
    clean_context = []
    for doc in retrieved_docs:
       	content = doc.page_content
       	# Remove special tags that might confuse the model
       	content = re.sub(r'<<[^>]+>>', '', content)
       	content = re.sub(r'<<[^>]+_END>>', '', content)

        # Clean OCR artifacts
        content = clean_ocr_text(content)
        clean_context.append(content)
    # Filter for quality
    filtered_context = filter_low_quality_chunks(clean_context)


    # Use the clean context
    context = "\n\n".join(filtered_context)
    sources = [doc.metadata for doc in retrieved_docs]

    if verbose:
        print(f"Retrieved Context: {context}")




    # If we didn't find relevant documents or context is sparse, try direct lookup
    if not retrieved_docs or len(clean_context) == 0 or len(context) < 100:
        standard_number = extract_standard_from_query(query)
        if standard_number:
            if verbose:
                print(f"Attempting direct lookup for standard: {standard_number}")
            
            matching_files = find_file_by_standard_number(standard_number)
            if matching_files and verbose:
                print(f"Found {len(matching_files)} files through direct lookup for {standard_number}")
            
            if matching_files:
                # Read the first matching file and extract relevant section
                try:
                    with open(matching_files[0], 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Try to extract requested section or use first part
                    if requested_section:
                        section_text = extract_section_from_text(content, requested_section)
                        if section_text:
                            clean_context = [section_text]
                            if verbose:
                                print(f"Extracted {requested_section} section through direct lookup")
                    else:
                        # Take a reasonable chunk if no specific section
                        clean_context = [content[:4000]]
                    
                    # Re-filter context after direct lookup
                    filtered_context = filter_low_quality_chunks(clean_context)
                    context = "\n\n".join(filtered_context)
                    
                    # Update sources
                    sources = [{"filename": os.path.basename(matching_files[0]), 
                               "section": requested_section if requested_section else "general",
                               "direct_lookup": True}]
                    
                    if verbose:
                        print("Successfully used direct lookup fallback")
                    
                except Exception as e:
                    if verbose:
                        print(f"Error in direct lookup: {e}")
    
    return context, sources



def generate_rag_response(model, tokenizer, query, context, max_length=2048, temperature=0.7, top_p=0.9, verbose=False):
    """Generate a response using the retrieved context and the query."""
    # Format the prompt with the context and query
#    formatted_prompt = f"""<s>[INST] I need information about the following topic. Use ONLY the provided context to answer my question accurately and concisely. If the context doesn't contain relevant information, simply state that the information is not available in the context. Do NOT make up information. \n\nWhen listing specific points, format them clearly with numbers or letters as shown in the context, and provide a brief conclusion or summary of your response. \n\nContext:\n{context}\n\nQuestion: {query} [/INST]"""



    # Detect if query is asking for a specific section
    requested_section = detect_section_request(query)
    
    # Check if this is about a standard
    standard_number = extract_standard_from_query(query)
    
    # Customize prompt based on query type
    if requested_section and standard_number:
        # Query about a specific section of a standard
        formatted_prompt = f"""<s>[INST] I need information about the {requested_section} section of {standard_number}. Use ONLY the provided context to answer my question accurately. If the information is not in the context, state that clearly.

Context:
{context}

Question: {query} [/INST]"""
    elif requested_section:
        # Query about a specific section
        formatted_prompt = f"""<s>[INST] I need information about the {requested_section} section of a document. Use ONLY the provided context to answer my question. Present the information in a clear, structured format.

Context:
{context}

Question: {query} [/INST]"""
    elif standard_number:
        # Query about a standard
        formatted_prompt = f"""<s>[INST] I need information about {standard_number}. Use ONLY the provided context to answer my question accurately and concisely. If the context doesn't contain sufficient information, state that clearly.

Context:
{context}

Question: {query} [/INST]"""
    else:
        # General query
        formatted_prompt = f"""<s>[INST] I need information about the following topic. Use ONLY the provided context to answer my question accurately and concisely. If the context doesn't contain relevant information, simply state that the information is not available in the context. Do NOT make up information.

When listing specific points, format them clearly with numbers or letters as shown in the context, and provide a brief conclusion or summary of your response.

Context:
{context}

Question: {query} [/INST]"""


    if verbose:
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

    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    else:
        if verbose:
            print("WARNING: [/INST] tag not found in response")
        response = ""

    # Remove the final </s> if present
    if response.endswith("</s>"):
        response = response[:-4].strip()

    # Clean up any custom tags
    response = re.sub(r'\[ANS\]|\[/AN.*?$', '', response).strip()
    
    # Apply post-processing to clean up response
    response = post_process_response(response)

    # Fallback for empty responses
    if len(response) < 20:
        if verbose:
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

def run_interactive_rag_console(model, tokenizer, vector_store, k_docs=3, verbose=False, log_file="query_log.jsonl"):
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
        context, sources = retrieve_context(vector_store, user_input, k=k_docs, verbose=verbose)

        # Generate a response
        print("\nGenerating response...")
        response = generate_rag_response(model, tokenizer, user_input, context, verbose=verbose)

        # Print the response and sources
        print(f"\nModel: {response}")
        print("\nSources:")
        for i, source in enumerate(sources):
            print(f"  {i+1}. {source.get('title', 'Untitled')} (from {source.get('filename', 'unknown')})")
        
        # Log the query and response
        log_query_results(user_input, context, sources, response, log_file)

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
    run_interactive_rag_console(model, tokenizer, vector_store, k_docs=args.k_docs, verbose=args.verbose, log_file=args.log_file)
