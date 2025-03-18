import os
import argparse
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

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

def retrieve_context(vector_store, query, k=3):
    """Retrieve the most relevant documents for the query."""
    print(f"Retrieving context for query: {query}")
    retrieved_docs = vector_store.similarity_search(query, k=k)
    
    print("Retrieved Docs:", retrieved_docs)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    sources = [doc.metadata for doc in retrieved_docs]
    
    return context, sources

def generate_rag_response(model, tokenizer, query, context, max_length=2048, temperature=0.7, top_p=0.9):
    """Generate a response using the retrieved context and the query."""
    # Format the prompt with the context and query
    formatted_prompt = f"<s>[INST] I need information about the following topic. Use the provided context to answer my question accurately.\n\nContext:\n{context}\n\nQuestion: {query} [/INST]"
    
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
    response = response.split("[/INST]")[-1].strip()
    
    # Remove the final </s> if present
    if response.endswith("</s>"):
        response = response[:-4].strip()
    
    return response

def run_interactive_rag_console(model, tokenizer, vector_store):
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
        context, sources = retrieve_context(vector_store, user_input)
        print("\nRetrieved Context:", context)


        # Generate a response
        print("\nGenerating response...")
        response = generate_rag_response(model, tokenizer, user_input, context)
        

        # Debugging: Check what response is generated
        print("\nDEBUG: Model Output:", response)



        # Print the response and sources
        print(f"\nModel: {response}")
        print("\nSources:")
        for i, source in enumerate(sources):
            print(f"  {i+1}. {source.get('title', 'Untitled')} (from {source.get('filename', 'unknown')})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG inference with the fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--vector_store_path", type=str, required=True, help="Path to the vector store")
    parser.add_argument("--device", type=str, default="auto", help="Device to run inference on")
    parser.add_argument("--use_lora", action="store_true", help="Whether the model is a LoRA model")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name if using LoRA")
    
    args = parser.parse_args()
    
    # Load the model
    model, tokenizer = load_model(args.model_path, args.device, args.use_lora, args.base_model)
    
    # Load the vector store
    vector_store = load_vector_store(args.vector_store_path)
    
    # Run the interactive console
    run_interactive_rag_console(model, tokenizer, vector_store)
