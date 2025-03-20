#!/usr/bin/env python3
import os
import sys
import argparse
import tempfile
import shutil
import json
import re
import datetime
import torch
from tqdm import tqdm
from pathlib import Path

# Import required libraries
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Try to import PDF processing libraries
try:
    import fitz  # PyMuPDF
    import pdfplumber
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    import io
    HAS_PDF_LIBS = True
except ImportError:
    HAS_PDF_LIBS = False
    print("Warning: PDF processing libraries not found. PDF processing will not be available.")

# Add project root to path for utility imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.query_utils import detect_file_references, build_metadata_filter, get_available_files
    from utils.context_analysis import rerank_documents
    from utils.hallucination_reduction import filter_low_quality_chunks, clean_ocr_text
    from utils.document_structure import detect_section_request, extract_section_from_text
    from utils.direct_lookup import extract_standard_from_query, find_file_by_standard_number
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False
    # Define minimal utility functions if imports fail
    # These functions provide basic functionality when the full utils aren't available
    def clean_ocr_text(text):
        """Basic OCR text cleaning."""
        text = re.sub(r'([.!?,;:]){2,}', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def filter_low_quality_chunks(chunks, threshold=0.3):
        """Simple low quality chunk filtering."""
        return [chunk for chunk in chunks if len(chunk.strip()) > 100]
    
    def detect_section_request(query):
        """Simple section detection."""
        section_keywords = {
            'scope': ['scope', 'purpose', 'objective'],
            'introduction': ['introduction', 'overview'],
            'requirements': ['requirements', 'specifications'],
            'conclusion': ['conclusion', 'summary']
        }
        
        query_lower = query.lower()
        for section, keywords in section_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return section
        return None
    
    def extract_section_from_text(text, section_name):
        """Extract section using pattern matching."""
        patterns = [
            rf'<<HEADING level=\d+>>{section_name}<<HEADING_END>>',
            rf'\d+\.\s+{section_name}',
            rf'{section_name.upper()}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_pos = match.start()
                next_section = re.search(r'<<HEADING level=\d+>>', text[start_pos+1:])
                if next_section:
                    end_pos = start_pos + 1 + next_section.start()
                    return text[start_pos:end_pos]
                else:
                    end_pos = min(start_pos + 2000, len(text))
                    return text[start_pos:end_pos]
        return None
    
    def detect_file_references(query):
        """Extract file references from query."""
        patterns = [r'file[s]?\s+(?:called|named)?\s+[\'"]?([^\'".,\s]+)[\'"]?']
        results = []
        for pattern in patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                results.append(match.group(1))
        return results
    
    def build_metadata_filter(file_references, available_files=None):
        """Build metadata filter from file references."""
        if not file_references:
            return None
        filters = []
        for ref in file_references:
            filters.append({"filename": {"$regex": ref}})
        return {"$or": filters} if len(filters) > 1 else filters[0]
    
    def extract_standard_from_query(query):
        """Extract standard numbers from query."""
        match = re.search(r'(?:ISO|IEC|BS)[- ]?(\d+(?:-\d+)?)', query)
        return match.group(0) if match else None
    
    def find_file_by_standard_number(standard_number, data_dir="data"):
        """Find files matching a standard number."""
        matching_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if standard_number in file and file.endswith('.txt'):
                    matching_files.append(os.path.join(root, file))
        return matching_files

# Define constants
TEMP_DIR = "temp_rag_files"
OCR_DPI = 300
MIN_TEXT_LENGTH = 50

# Document tag markers
LLM_MARKERS = {
    "document_start": "<<DOCUMENT_START>>",
    "document_end": "<<DOCUMENT_END>>",
    "page_start": "<<PAGE_{}>>",
    "page_end": "<<PAGE_END>>",
    "section_start": "<<SECTION_START:{}>>",
    "section_end": "<<SECTION_END>>",
    "heading": "<<HEADING level={}>>{}<<HEADING_END>>",
    "paragraph": "<<PARAGRAPH>>{}<<PARAGRAPH_END>>",
}

def is_pdf(file_path):
    """Check if file is a PDF."""
    return file_path.lower().endswith('.pdf')

def is_tagged_text(file_path):
    """Check if file is in tagged text format."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(1000)
            return '<<DOCUMENT_START>>' in content
    except:
        return False

def is_processed_jsonl(file_path):
    """Check if file is a processed JSONL file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            return first_line.startswith('{') and '"content":' in first_line
    except:
        return False

def extract_pdf_to_tagged_text(pdf_path, output_path, verbose=False):
    """Extract text from PDF and format with document tags."""
    if verbose:
        print(f"Extracting text from PDF: {pdf_path}")
    
    if not HAS_PDF_LIBS:
        raise ImportError("PDF processing libraries not available")
    
    # Check if scanned or digital
    is_scanned_doc = is_scanned(pdf_path)
    if verbose:
        print(f"Document type: {'Scanned' if is_scanned_doc else 'Digital'}")
    
    # Process PDF based on type
    if is_scanned_doc:
        page_data_list = extract_with_ocr(pdf_path)
        tables_data = []
    else:
        page_data_list, tables_data, _ = extract_native_text(pdf_path)
    
    # Format text for LLM consumption
    llm_text = format_for_llm(page_data_list, tables_data, is_scanned_doc)
    
    # Save the formatted text
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(llm_text)
    
    if verbose:
        print(f"Extracted text saved to {output_path}")
    
    return output_path

def is_scanned(pdf_path, sample_pages=5):
    """Determine if PDF is scanned using text length heuristic."""
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    pages_to_check = range(min(num_pages, sample_pages))
    scanned_page_count = 0
    
    for page_num in pages_to_check:
        page = doc[page_num]
        text = page.get_text("text")
        if len(text.strip()) < MIN_TEXT_LENGTH:
            scanned_page_count += 1
    
    return scanned_page_count > (len(pages_to_check) // 2)

def extract_with_ocr(pdf_path):
    """Extract text using OCR for scanned documents."""
    images = convert_from_path(pdf_path, dpi=OCR_DPI)
    page_data_list = []
    
    for page_num, image in enumerate(tqdm(images, desc="OCR Processing")):
        text = pytesseract.image_to_string(image)
        paragraphs = text.split("\n\n")
        
        page_data = {
            "page_num": page_num + 1,
            "blocks": [
                {
                    "type": "text",
                    "content": para.strip(),
                    "block_id": f"b{page_num+1}_{i}"
                }
                for i, para in enumerate(paragraphs) if para.strip()
            ]
        }
        page_data_list.append(page_data)
    
    return page_data_list

def extract_native_text(pdf_path):
    """Extract text from digital PDFs."""
    page_data_list = []
    tables_data = []
    annotations_data = []
    
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    
    for page_num in tqdm(range(num_pages), desc="Extracting text"):
        page = doc[page_num]
        page_data = {"page_num": page_num + 1, "blocks": []}
        
        # Extract blocks from page
        page_dict = page.get_text("dict")
        for block_idx, block in enumerate(page_dict["blocks"]):
            if block["type"] == 0:  # Text block
                block_text = []
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"] + " "
                    if line_text.strip():
                        block_text.append(line_text.strip())
                
                if block_text:
                    # Determine if heading based on font size
                    font_size = 0
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_size = max(font_size, span["size"])
                    
                    is_heading = font_size >= 12 and len("\n".join(block_text)) < 200
                    
                    page_data["blocks"].append({
                        "type": "text",
                        "content": "\n".join(block_text),
                        "is_heading": is_heading,
                        "font_size": font_size
                    })
        
        page_data_list.append(page_data)
        
        # Extract tables using pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    tables = page.extract_tables()
                    for i, table in enumerate(tables):
                        if table:
                            tables_data.append({
                                "table_id": f"table_{page_num+1}_{i+1}",
                                "page": page_num + 1,
                                "data": table
                            })
        except:
            pass
    
    return page_data_list, tables_data, annotations_data

def format_for_llm(page_data_list, tables_data, is_scanned):
    """Format extracted data for LLM consumption."""
    llm_content = [LLM_MARKERS["document_start"]]
    
    # Process each page
    for page_data in page_data_list:
        page_num = page_data["page_num"]
        llm_content.append(LLM_MARKERS["page_start"].format(page_num))
        
        # Process blocks
        for block in page_data["blocks"]:
            if block["type"] == "text":
                if block.get("is_heading", False):
                    # Estimate heading level based on font size
                    level = 1 if block.get("font_size", 0) >= 16 else 2 if block.get("font_size", 0) >= 14 else 3
                    llm_content.append(LLM_MARKERS["heading"].format(level, block["content"]))
                else:
                    llm_content.append(LLM_MARKERS["paragraph"].format(block["content"]))
        
        # Add tables for this page
        page_tables = [table for table in tables_data if table["page"] == page_num]
        for table in page_tables:
            table_str = f"<<TABLE_{table['table_id'].split('_')[-1]}>>\n"
            for row in table["data"]:
                table_str += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
            table_str += "<<TABLE_END>>"
            llm_content.append(table_str)
        
        llm_content.append(LLM_MARKERS["page_end"])
    
    # Add metadata section
    llm_content.append(LLM_MARKERS["section_start"].format("METADATA"))
    llm_content.append(f"Document Type: {'Scanned' if is_scanned else 'Digital'}")
    llm_content.append(f"Total Pages: {len(page_data_list)}")
    if tables_data:
        llm_content.append(f"Total Tables: {len(tables_data)}")
    llm_content.append(LLM_MARKERS["section_end"])
    llm_content.append(LLM_MARKERS["document_end"])
    
    return "\n".join(llm_content)

def preprocess_tagged_text(text_file_path, verbose=False):
    """Process text files with special tags into structured format."""
    if verbose:
        print(f"Processing tagged text: {text_file_path}")
    
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
        return '#' * int(level) + ' ' + text.strip()
    
    content = re.sub(r'<<HEADING level=(\d+)>>(.*?)<<HEADING_END>>', heading_replacer, content)
    
    # Process paragraphs
    content = re.sub(r'<<PARAGRAPH>>', '', content)
    content = re.sub(r'<<PARAGRAPH_END>>', '\n\n', content)
    
    # Remove section markers
    content = re.sub(r'<<SECTION_START:[^>]+>>', '', content)
    content = re.sub(r'<<SECTION_END>>', '', content)
    
    # Clean up whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.strip()
    
    return {
        'content': content,
        'metadata': metadata
    }

def save_processed_to_jsonl(processed_data, output_file):
    """Save processed document data to JSONL format."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(processed_data) + '\n')
    
    return output_file

def build_vector_store(jsonl_file, output_dir, chunk_size=1000, chunk_overlap=200, verbose=False):
    """Build a vector store from processed JSONL data."""
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("Initializing embedding model...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
    )
    
    documents = []
    
    if verbose:
        print("Loading document and splitting into chunks...")
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
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
                        'chunk_id': i,
                        'source_path': metadata.get('source_path', '')
                    }
                )
                documents.append(doc)
    
    if verbose:
        print(f"Created {len(documents)} document chunks for vectorization")
    
    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    
    if verbose:
        print(f"Saving vector store to {output_dir}")
    
    vector_store.save_local(output_dir)
    
    return vector_store

def load_model(model_path, device="auto", use_lora=True, base_model=None, verbose=False):
    """Load the language model."""
    if use_lora:
        if base_model is None:
            try:
                config = PeftConfig.from_pretrained(model_path)
                base_model = config.base_model_name_or_path
            except:
                raise ValueError("Base model must be provided for LoRA models")
        
        if verbose:
            print(f"Loading base model: {base_model}")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        if verbose:
            print(f"Loading LoRA adapter: {model_path}")
        
        model = PeftModel.from_pretrained(model, model_path)
    else:
        if verbose:
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

def retrieve_context(vector_store, query, k=3, verbose=False):
    """Retrieve relevant context from the vector store."""
    if verbose:
        print(f"Retrieving context for query: {query}")
    
    # Try to get available files
    try:
        available_files = get_available_files(vector_store) if HAS_UTILS else []
    except:
        available_files = []
    
    # Detect requested section
    requested_section = detect_section_request(query) if HAS_UTILS else None
    
    # Detect file references
    file_references = detect_file_references(query) if HAS_UTILS else []
    
    # Build metadata filter
    metadata_filter = None
    if HAS_UTILS and file_references and available_files:
        metadata_filter = build_metadata_filter(file_references, available_files)
    
    # Try metadata-filtered search first
    retrieved_docs = None
    if metadata_filter:
        try:
            retrieved_docs = vector_store.similarity_search(
                query, k=k*3, filter=metadata_filter
            )
        except Exception as e:
            if verbose:
                print(f"Error with metadata filtering: {e}")
    
    # Fall back to regular search
    if not retrieved_docs:
        retrieved_docs = vector_store.similarity_search(query, k=k*3)
    
    # Process section requests if available
    if HAS_UTILS and retrieved_docs and requested_section:
        # Group by filename
        docs_by_filename = {}
        for doc in retrieved_docs:
            filename = doc.metadata.get('filename', 'unknown')
            if filename not in docs_by_filename:
                docs_by_filename[filename] = []
            docs_by_filename[filename].append(doc.page_content)
        
        # Extract sections
        section_contexts = []
        for filename, contents in docs_by_filename.items():
            full_text = "\n\n".join(contents)
            section_text = extract_section_from_text(full_text, requested_section)
            if section_text:
                section_contexts.append((filename, section_text))
        
        # Create section documents
        if section_contexts:
            section_docs = []
            for filename, section_text in section_contexts:
                clean_text = re.sub(r'<<[^>]+>>', '', section_text)
                clean_text = re.sub(r'<<[^>]+_END>>', '', clean_text)
                
                meta = {"filename": filename, "section": requested_section}
                section_docs.append(Document(page_content=clean_text, metadata=meta))
            
            retrieved_docs = section_docs[:k]
    
    # Clean up and filter context
    clean_context = []
    for doc in retrieved_docs[:k]:
        content = doc.page_content
        
        # Remove special tags
        content = re.sub(r'<<[^>]+>>', '', content)
        content = re.sub(r'<<[^>]+_END>>', '', content)
        
        # Clean OCR artifacts if available
        if HAS_UTILS:
            content = clean_ocr_text(content)
        
        clean_context.append(content)
    
    # Filter for quality
    if HAS_UTILS:
        filtered_context = filter_low_quality_chunks(clean_context)
    else:
        filtered_context = clean_context
    
    # Join context
    context = "\n\n".join(filtered_context)
    sources = [doc.metadata for doc in retrieved_docs[:k]]
    
    return context, sources

def generate_response(model, tokenizer, query, context, max_length=2048, temperature=0.7, verbose=False):
    """Generate a response using the LLM."""
    # Create the prompt for the model
    formatted_prompt = f"""<s>[INST] I need information about the following topic. Use ONLY the provided context to answer my question accurately. If the information is not available in the context, state that clearly. Do not make up information.

Context:
{context}

Question: {query} [/INST]"""
    
    if verbose:
        print(f"Prompt length: {len(formatted_prompt)}")
    
    # Generate response
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
    
    # Process response
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    
    if response.endswith("</s>"):
        response = response[:-4].strip()
    
    return response

def run_rag_console(model, tokenizer, vector_store, k_docs=3, verbose=False, log_file="query_log.jsonl"):
    """Run interactive RAG console for document."""
    print("\n📚 Document Assistant")
    print("Ask questions about this document. Type 'exit' to quit.")
    print("-----------------------------------------------------")
    
    try:
        while True:
        # Get user query
            user_input = input("\nQuestion: ")
        
        # Check for exit
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting. Goodbye!")
                break
        
        # Retrieve context
            context, sources = retrieve_context(vector_store, user_input, k=k_docs, verbose=verbose)
        
        # Generate response
            print("\nGenerating response...")
            response = generate_response(model, tokenizer, user_input, context, verbose=verbose)
        
            # Display response and sources
            print(f"\nAnswer: {response}")
            print("\nSources:")


            unique_sources = set()

            for source in sources:
                unique_sources.add((source.get('title', 'Untitled'), source.get('filename', 'unknown')))
            for i, (title, filename) in enumerate(unique_sources, 1):
                print(f"  {i}. {title} (from {filename})")

            #for i, source in enumerate(sources):
             #   print(f"  {i+1}. {source.get('title', 'Untitled')} (from {source.get('filename', 'unknown')})")
        
        # Log the interaction
            if log_file:
                log_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "query": user_input,
                    "response": response,
                    "sources": [s.get("filename", "unknown") for s in sources]
                }
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
    finally:
        temp_dir="temp_rag_files"
        print("Cleaning up temporary files...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print("Cleanup complete. Exiting.")
def process_input_file(input_path, verbose=False):
    """Process input file into JSONL format."""
    temp_dir = os.path.join(TEMP_DIR, "single_doc_rag")
    os.makedirs(temp_dir, exist_ok=True)
    
    base_name = os.path.basename(input_path).split('.')[0]
    
    # Handle different input types
    if is_pdf(input_path):
        # Process PDF to tagged text
        tagged_text_path = os.path.join(temp_dir, f"{base_name}.txt")
        extract_pdf_to_tagged_text(input_path, tagged_text_path, verbose)
        
        # Process tagged text to JSONL
        doc_data = preprocess_tagged_text(tagged_text_path, verbose)
        jsonl_path = os.path.join(temp_dir, f"{base_name}.jsonl")
        save_processed_to_jsonl(doc_data, jsonl_path)
    
    elif is_tagged_text(input_path):
        # Process tagged text to JSONL
        doc_data = preprocess_tagged_text(input_path, verbose)
        jsonl_path = os.path.join(temp_dir, f"{base_name}.jsonl")
        save_processed_to_jsonl(doc_data, jsonl_path)
    
    elif is_processed_jsonl(input_path):
        # Already in JSONL format
        jsonl_path = input_path
    
    else:
        # Treat as plain text
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create minimal tagged structure
        tagged_content = f"{LLM_MARKERS['document_start']}\n"
        tagged_content += f"{LLM_MARKERS['page_start'].format(1)}\n"
        tagged_content += f"{LLM_MARKERS['heading'].format(2, os.path.basename(input_path))}\n"
        tagged_content += f"{LLM_MARKERS['paragraph'].format(content)}\n"
        tagged_content += f"{LLM_MARKERS['page_end']}\n"
        tagged_content += f"{LLM_MARKERS['section_start'].format('METADATA')}\n"
        tagged_content += f"Document Type: Text\nTotal Pages: 1\n"
        tagged_content += f"{LLM_MARKERS['section_end']}\n"
        tagged_content += f"{LLM_MARKERS['document_end']}\n"
        
        tagged_text_path = os.path.join(temp_dir, f"{base_name}.txt")
        with open(tagged_text_path, 'w', encoding='utf-8') as f:
            f.write(tagged_content)
        
        doc_data = preprocess_tagged_text(tagged_text_path, verbose)
        jsonl_path = os.path.join(temp_dir, f"{base_name}.jsonl")
        save_processed_to_jsonl(doc_data, jsonl_path)
    
    return jsonl_path

def main():
    """Main function for single document RAG."""
    parser = argparse.ArgumentParser(description="Single Document RAG System")
    parser.add_argument("--input_path", type=str, required=True, 
                      help="Path to input file (PDF or text)")
    parser.add_argument("--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", 
                      help="Path to model")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA adapter")
    parser.add_argument("--base_model", type=str, default=None, help="Base model for LoRA")
    parser.add_argument("--vector_db_path", type=str, help="Vector database path")
    parser.add_argument("--k_docs", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--log_file", type=str, default="single_doc_queries.jsonl", 
                      help="Path to log file")
    parser.add_argument("--clean_up", action="store_true", help="Clean temporary files after running")
    
    args = parser.parse_args()
    
    try:
        # Create temp directory
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Process input file
        jsonl_path = process_input_file(args.input_path, args.verbose)
        
        # Set vector database path
        vector_db_path = args.vector_db_path or os.path.join(TEMP_DIR, "vector_db")
        
        # Build vector store
        vector_store = build_vector_store(jsonl_path, vector_db_path, verbose=args.verbose)
        
        # Load model
        model, tokenizer = load_model(args.model_path, use_lora=args.use_lora, 
                                    base_model=args.base_model, verbose=args.verbose)
        
        # Run interactive console
        run_rag_console(model, tokenizer, vector_store, k_docs=args.k_docs,
                      verbose=args.verbose, log_file=args.log_file)
        
        # Clean up if requested
        if args.clean_up and os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
    
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
