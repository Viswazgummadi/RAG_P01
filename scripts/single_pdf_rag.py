import os
import sys
import argparse
import torch
import re
import tempfile
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PyPDF2 import PdfReader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hallucination_reduction import filter_low_quality_chunks, clean_ocr_text
from utils.document_structure import detect_section_request, extract_section_from_text

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    print(f"Extracting text from PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    return text

def create_vector_store_from_pdf(pdf_path, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """Create a vector store from a single PDF file."""
    print(f"Creating vector store from: {pdf_path}")
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", ";", ",", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    print(f"Split PDF into {len(chunks)} text chunks")
    
    # Create metadata for each chunk
    pdf_name = os.path.basename(pdf_path)
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "filename": pdf_name,
                "source": pdf_path,
                "chunk_id": i
            }
        ) for i, chunk in enumerate(chunks)
    ]
    
    # Create vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def load_model(model_path, device="auto", use_lora=True, base_model=None):
    """Load the language model."""
    # Reusing your existing load_model function
    if use_lora:
        if base_model is None:
            try:
                config = PeftConfig.from_pretrained(model_path)
                base_model = config.base_model_name_or_path
            except:
                raise ValueError("Base model must be provided for LoRA models")
                
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

def retrieve_context_from_pdf(vector_store, query, pdf_name, k=3, verbose=False):
    """Retrieve context from the single PDF vector store."""
    if verbose:
        print(f"Retrieving context for query: {query}")
    
    # Check for section requests
    requested_section = detect_section_request(query)
    if verbose and requested_section:
        print(f"Detected section request: {requested_section}")
    
    # Retrieve relevant chunks
    k_retrieval = min(k * 2, 10)
    retrieved_docs = vector_store.similarity_search(query, k=k_retrieval)
    
    # Clean up chunks
    clean_context = []
    for doc in retrieved_docs:
        content = doc.page_content
        # Apply cleaning if needed
        content = clean_ocr_text(content)
        clean_context.append(content)
    
    # Filter for quality
    filtered_context = filter_low_quality_chunks(clean_context)
    
    # Join for final context
    context = "\n\n".join(filtered_context)
    sources = [{"filename": pdf_name, "chunk_id": doc.metadata.get("chunk_id", "unknown")} 
              for doc in retrieved_docs[:k]]
    
    return context, sources

def generate_pdf_response(model, tokenizer, query, context, pdf_name, max_length=2048, 
                         temperature=0.7, top_p=0.9, verbose=False):
    """Generate a response using the single PDF context."""
    # Creating a specialized prompt for single PDF queries
    formatted_prompt = f"""<s>[INST] I'm analyzing a document named "{pdf_name}". Answer the following question using ONLY information found in the document extracts provided below.

If the information needed to answer the question is NOT present in these extracts, clearly state: "I cannot find information about this in the document." DO NOT make up information or use knowledge outside this document.

Document extracts:
{context}

Question: {query} [/INST]"""

    if verbose:
        print(f"Prompt length: {len(formatted_prompt)}")
    
    # Tokenize and generate
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
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
    
    # Process response
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    
    if response.endswith("</s>"):
        response = response[:-4].strip()
    
    return response

def run_pdf_assistant(model, tokenizer, vector_store, pdf_name, verbose=False):
    """Run an interactive PDF assistant loop."""
    print(f"\n📄 PDF Assistant: {pdf_name}")
    print("Ask questions about this PDF document. Type 'exit' to quit.")
    print("-----------------------------------------------------")
    
    while True:
        # Get user question
        user_input = input("\nQuestion: ")
        
        # Check for exit
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting PDF Assistant. Goodbye!")
            break
        
        # Retrieve context
        context, sources = retrieve_context_from_pdf(
            vector_store, 
            user_input, 
            pdf_name, 
            verbose=verbose
        )
        
        if verbose:
            print(f"\nRetrieved Context: {context[:300]}...")
        
        # Generate response
        print("\nGenerating response...")
        response = generate_pdf_response(
            model, 
            tokenizer, 
            user_input, 
            context, 
            pdf_name, 
            verbose=verbose
        )
        
        # Display response
        print(f"\nAnswer: {response}")
        
        # Show source information
        print(f"\nSource: {pdf_name} (Using {len(sources)} relevant sections)")

def main():
    parser = argparse.ArgumentParser(description="PDF Document Assistant using RAG")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", 
                      help="Path to the model or model name")
    parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA adapter")
    parser.add_argument("--base_model", type=str, default=None, 
                      help="Base model name if using LoRA")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Check if PDF exists
    if not os.path.exists(args.pdf_path) or not args.pdf_path.lower().endswith('.pdf'):
        print(f"Error: {args.pdf_path} is not a valid PDF file")
        return
    
    # Create vector store from PDF
    vector_store = create_vector_store_from_pdf(args.pdf_path)
    
    # Load model
    model, tokenizer = load_model(
        args.model_path, 
        use_lora=args.use_lora, 
        base_model=args.base_model
    )
    
    # Get PDF name for display
    pdf_name = os.path.basename(args.pdf_path)
    
    # Run the assistant
    run_pdf_assistant(model, tokenizer, vector_store, pdf_name, verbose=args.verbose)

if __name__ == "__main__":
    main()
