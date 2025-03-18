import os
import time
import argparse
from preprocess import process_directory
from vector_db import build_vector_store
from training_data import create_training_data
from train import finetune_mistral
from query_system import PDFMemorySystem
from interface import interactive_cli, create_web_ui

def main():
    parser = argparse.ArgumentParser(description="PDF Memory System")
    parser.add_argument("--task", type=str, choices=[
        "preprocess", "vectorize", "create_training", "train", "query", "all"
    ], required=True, help="Task to perform")
    
    parser.add_argument("--input_dir", type=str, help="Directory containing text files")
    parser.add_argument("--output_dir", type=str, default="./pdf_memory_system", 
                        help="Output directory for all data and models")
    
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", 
                        help="Hugging Face model ID for Mistral")
    
    parser.add_argument("--interface", type=str, choices=["cli", "web"], default="cli", 
                        help="Interface type for querying")
    
    parser.add_argument("--port", type=int, default=7860, help="Port for web interface")
    
    parser.add_argument("--samples_per_doc", type=int, default=5, 
                        help="Number of training samples per document")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define paths
    jsonl_path = os.path.join(args.output_dir, "processed_data.jsonl")
    vector_db_path = os.path.join(args.output_dir, "vector_db")
    training_data_path = os.path.join(args.output_dir, "training_data")
    model_path = os.path.join(args.output_dir, "fine_tuned_model", "final")
    
    # Execute the selected task
    if args.task == "preprocess" or args.task == "all":
        if not args.input_dir:
            raise ValueError("--input_dir must be specified for preprocess task")
        
        print("Starting preprocessing...")
        start_time = time.time()
        process_directory(args.input_dir, jsonl_path)
        print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    
    if args.task == "vectorize" or args.task == "all":
        if not os.path.exists(jsonl_path):
            raise ValueError(f"JSONL file not found at {jsonl_path}. Run preprocess first.")
        
        print("Starting vector database creation...")
        start_time = time.time()
        build_vector_store(jsonl_path, vector_db_path)
        print(f"Vector database creation completed in {time.time() - start_time:.2f} seconds")
    
    if args.task == "create_training" or args.task == "all":
        if not os.path.exists(jsonl_path):
            raise ValueError(f"JSONL file not found at {jsonl_path}. Run preprocess first.")
        
        print("Creating training data...")
        start_time = time.time()
        create_training_data(jsonl_path, training_data_path, args.samples_per_doc)
        print(f"Training data creation completed in {time.time() - start_time:.2f} seconds")
    
    if args.task == "train" or args.task == "all":
        if not os.path.exists(training_data_path):
            raise ValueError(f"Training data not found at {training_data_path}. Run create_training first.")
        
        print("Starting model fine-tuning...")
        start_time = time.time()
        finetune_mistral(training_data_path, os.path.join(args.output_dir, "fine_tuned_model"), args.model_id)
        print(f"Model fine-tuning completed in {time.time() - start_time:.2f} seconds")
    
    if args.task == "query" or args.task == "all":
        if not os.path.exists(model_path):
            raise ValueError(f"Fine-tuned model not found at {model_path}. Run train first.")
        
        if not os.path.exists(vector_db_path):
            raise ValueError(f"Vector database not found at {vector_db_path}. Run vectorize first.")
        
        print("Initializing PDF Memory System...")
        pdf_memory = PDFMemorySystem(model_path, vector_db_path)
        
        if args.interface == "cli":
            interactive_cli(pdf_memory)
        else:
            create_web_ui(pdf_memory, args.port)

if __name__ == "__main__":
    main()
