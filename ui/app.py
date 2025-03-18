import os
import argparse
import sys
import gradio as gr
import torch

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import load_model_and_tokenizer, generate_text
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_vector_store(vector_store_path):
    """Load the vector store from the given path."""
    print(f"Loading vector store from {vector_store_path}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    vector_store = FAISS.load_local(vector_store_path, embeddings)
    return vector_store

def retrieve_context(vector_store, query, k=3):
    """Retrieve the most relevant documents for the query."""
    retrieved_docs = vector_store.similarity_search(query, k=k)
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    sources = [doc.metadata for doc in retrieved_docs]
    
    return context, sources

def create_ui(model, tokenizer, vector_store):
    """Create the Gradio UI for interacting with the model."""
    # Define the pure model interface
    def pure_model_fn(query, temperature, max_tokens):
        # Format the prompt according to the Mistral chat template
        formatted_prompt = f"<s>[INST] {query} [/INST]"
        
        response = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_length=max_tokens,
            temperature=temperature
        )
        
        return response
    
    # Define the RAG model interface
    def rag_model_fn(query, temperature, max_tokens, k_docs):
        # Retrieve context
        context, sources = retrieve_context(vector_store, query, k=k_docs)
        
        # Format the prompt with the context and query
        formatted_prompt = f"<s>[INST] I need information about the following topic. Use the provided context to answer my question accurately.\n\nContext:\n{context}\n\nQuestion: {query} [/INST]"
        
        response = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_length=max_tokens,
            temperature=temperature
        )
        
        # Format sources for display
        sources_text = "Sources:\n"
        for i, source in enumerate(sources):
            title = source.get('title', 'Untitled')
            filename = source.get('filename', 'unknown')
            sources_text += f"{i+1}. {title} (from {filename})\n"
        
        return response, sources_text
    
    # Create the interface
    with gr.Blocks(title="Mistral Document Assistant") as demo:
        gr.Markdown("# Mistral Document Assistant")
        gr.Markdown("This application allows you to interact with a fine-tuned Mistral model to answer questions about your documents.")
        
        with gr.Tab("Standard Model"):
            gr.Markdown("Ask questions and get answers directly from the fine-tuned model.")
            
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(label="Your Question", placeholder="Ask a question...", lines=3)
                    with gr.Row():
                        temp_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature")
                        max_tokens_slider = gr.Slider(minimum=100, maximum=4000, value=1000, step=100, label="Max Tokens")
                    submit_btn = gr.Button("Submit")
                
                with gr.Column():
                    response_output = gr.Textbox(label="Model Response", lines=15)
            
            submit_btn.click(
                fn=pure_model_fn,
                inputs=[query_input, temp_slider, max_tokens_slider],
                outputs=response_output
            )
        
        with gr.Tab("RAG Model"):
            gr.Markdown("Ask questions with document retrieval for context-aware answers.")
            
            with gr.Row():
                with gr.Column():
                    rag_query_input = gr.Textbox(label="Your Question", placeholder="Ask a question...", lines=3)
                    with gr.Row():
                        rag_temp_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature")
                        rag_max_tokens_slider = gr.Slider(minimum=100, maximum=4000, value=1000, step=100, label="Max Tokens")
                        k_docs_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of Documents")
                    rag_submit_btn = gr.Button("Submit")
                
                with gr.Column():
                    rag_response_output = gr.Textbox(label="Model Response", lines=10)
                    sources_output = gr.Textbox(label="Source Documents", lines=5)
            
            rag_submit_btn.click(
                fn=rag_model_fn,
                inputs=[rag_query_input, rag_temp_slider, rag_max_tokens_slider, k_docs_slider],
                outputs=[rag_response_output, sources_output]
            )
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mistral Document Assistant UI")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--vector_store_path", type=str, required=True, help="Path to the vector store")
    parser.add_argument("--use_lora", action="store_true", help="Whether the model is a LoRA model")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name if using LoRA")
    parser.add_argument("--server_port", type=int, default=7860, help="Port to run the Gradio server on")
    
    args = parser.parse_args()
    
    # Load the model
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model_path,
        use_4bit=True,
        device="auto",
        bf16=True,
        is_adapter_path=args.use_lora,
        base_model=args.base_model
    )
    
    # Load the vector store
    vector_store = load_vector_store(args.vector_store_path)
    
    # Create and launch the UI
    demo = create_ui(model, tokenizer, vector_store)
    demo.launch(server_name="0.0.0.0", server_port=args.server_port, share=False)
