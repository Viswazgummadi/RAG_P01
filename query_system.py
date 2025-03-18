import os
import torch
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

class PDFMemorySystem:
    def __init__(self, model_path, vector_db_path, device=None):
        """Initialize the PDF Memory System."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing PDF Memory System on {device}...")
        
        # Initialize embedding model for vector search
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device if device != 'auto' else None}
        )
        
        # Load vector database
        print("Loading vector database...")
        self.vector_db = FAISS.load_local(vector_db_path, self.embeddings)
        print(f"Vector database loaded with {self.vector_db.index.ntotal} vectors")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if this is a PEFT model
        is_peft_model = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        # Load model based on type
        print("Loading model...")
        if is_peft_model:
            print("Detected PEFT model. Loading base model and adapters...")
            base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16,
                device_map=device
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            print("Loading regular model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device
            )
        
        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        print("PDF Memory System initialized and ready for queries")
    
    def query(self, question, num_chunks=5):
        """Query the PDF Memory System."""
        # Retrieve relevant document chunks
        results = self.vector_db.similarity_search(question, k=num_chunks)
        
        # Extract content and metadata
        contexts = []
        sources = []
        
        for doc in results:
            contexts.append(doc.page_content)
            source_info = {
                'title': doc.metadata.get('title', 'Untitled'),
                'filename': doc.metadata.get('filename', 'Unknown'),
                'chunk_id': doc.metadata.get('chunk_id', 0)
            }
            sources.append(source_info)
        
        # Create prompt for the model following Mistral's format
        prompt = f"""<s>[INST] Answer the following question based only on the provided context.
Include source information like document title and filename at the end of your answer.

Question: {question}

Context:
{' '.join(contexts)}
[/INST]"""
        
        # Generate answer
        response = self.generator(prompt)[0]['generated_text']
        
        # Extract the model's answer (remove the instruction)
        answer = response.split("[/INST]")[-1].strip()
        
        # Add source attribution if not included
        if not any(source['filename'] in answer for source in sources):
            answer += "\n\nSources:"
            unique_sources = {}
            for source in sources:
                key = f"{source['title']}_{source['filename']}"
                if key not in unique_sources:
                    unique_sources[key] = source
                    
            for source in unique_sources.values():
                answer += f"\n- {source['title']} ({source['filename']})"
        
        return answer
    
    def save_conversation(self, question, answer, output_file):
        """Save conversation to file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"Question: {question}\n\n")
            f.write(f"Answer: {answer}\n\n")
            f.write("-" * 50 + "\n\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize PDF Memory System for testing")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--vector_db_path", type=str, required=True, help="Path to vector database")
    parser.add_argument("--question", type=str, help="Test question (optional)")
    
    args = parser.parse_args()
    
    system = PDFMemorySystem(args.model_path, args.vector_db_path)
    
    if args.question:
        answer = system.query(args.question)
        print(f"Question: {args.question}\n")
        print(f"Answer: {answer}")
