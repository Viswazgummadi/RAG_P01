# Retrieval-Augmented Generation (RAG) System for Technical Document Analysis

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for analyzing technical documents, particularly standards and specifications. It integrates advanced NLP techniques, including document preprocessing, vector database creation, and fine-tuned language model inference, to generate accurate and context-aware responses to user queries.

## System Architecture
The RAG system consists of the following key components:

- **Document Preprocessing**
- **Vector Database Creation**
- **Language Model Fine-tuning**
- **Query Processing and Retrieval**
- **Response Generation**

## Implementation Details

### Document Preprocessing
- **PDF Processing:**
  - Utilizes `PyMuPDF` and `pdf2image` for text and image extraction.
  - Implements OCR using `Tesseract` for scanned documents.
  - Extracts metadata, text content, and structural information.
- **Text Formatting:**
  - Converts extracted content to a standardized format with special tags.
  - Preserves document structure (headings, paragraphs, tables).
- **Metadata Extraction:**
  - Captures document properties (title, author, date).
  - Identifies sections and their hierarchies.

### Vector Database Creation
- **Text Chunking:**
  - Implements `RecursiveCharacterTextSplitter` for content segmentation.
  - Optimizes chunk size and overlap for retrieval efficiency.
- **Embedding Generation:**
  - Uses `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2`.
  - Generates vector representations for each text chunk.
- **FAISS Integration:**
  - Implements FAISS for efficient similarity search.
  - Stores vectors and associated metadata for quick retrieval.

### Language Model Fine-tuning
- **Model Selection:** Uses `Mistral-7B-Instruct-v0.2` as the base model.
- **Training Data Preparation:**
  - Generates instruction-response pairs from processed documents.
  - Implements data cleaning and quality filtering.
- **Fine-tuning Process:**
  - Utilizes `PyTorch` and `Transformers` libraries.
  - Implements gradient accumulation and mixed-precision training.

### Query Processing and Retrieval
- **Query Analysis:**
  - Detects file references and section requests in user queries.
  - Implements metadata filtering for targeted retrieval.
- **Context Retrieval:**
  - Performs similarity search using FAISS.
  - Implements a fallback mechanism for direct lookup of standards.
- **Reranking:**
  - Implements a custom reranking algorithm for relevance improvement.
  - Considers document structure and query specificity.

### Response Generation
- **Prompt Engineering:**
  - Designs context-aware prompts for different query types.
  - Incorporates retrieved context and query information.
- **Inference:**
  - Utilizes the fine-tuned model for response generation.
  - Implements temperature and top-p sampling for controlled output.
- **Post-processing:**
  - Cleans generated responses to remove artifacts.
  - Implements fallback mechanisms for low-quality responses.

## Evaluation and Results
The system's performance was evaluated based on:
- **Retrieval Accuracy:** Measured by the relevance of retrieved contexts.
- **Response Quality:** Assessed via human evaluation and automated metrics (BLEU, ROUGE).
- **Inference Speed:** Benchmarked for different query types and document sizes.

## Future Work
Potential enhancements include:
- Multi-modal RAG with image analysis.
- Improved handling of mathematical and chemical formulas.
- Integration with domain-specific knowledge bases.

## Usage
```sh
# Preprocess documents
python preprocess.py --input_dir data2 --output_file processed_data/all_processed.jsonl

# Create vector database
python vector_db.py --jsonl_file processed_data/all_processed.jsonl --output_dir vector_db

# Run interactive RAG system
python scripts/inference_rag.py --model_path models/instruct_model/final \
  --vector_store_path vector_db --use_lora --base_model "mistralai/Mistral-7B-Instruct-v0.2"
```

## Requirements
- Python 3.8+
- PyTorch
- Transformers
- LangChain
- FAISS

## License
This project is licensed under the MIT License.
