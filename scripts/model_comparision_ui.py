import gradio as gr
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import re

def load_model(model_path, use_lora=False, base_model=None):
    """Load model with optional LoRA adapter."""
    if use_lora:
        if base_model is None:
            config = PeftConfig.from_pretrained(model_path)
            base_model = config.base_model_name_or_path
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_vector_store(vector_store_path):
    """Load vector store for RAG."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Global variables to store models and vector store
models = {}
vector_store = None

def setup_models(args):
    """Setup models and vector store."""
    global models, vector_store
    
    # Load base model
    print(f"Loading base model: {args.base_model}")
    base_model, base_tokenizer = load_model(args.base_model)
    models["base"] = {"model": base_model, "tokenizer": base_tokenizer}
    
    # Load fine-tuned model
    print(f"Loading fine-tuned model: {args.finetuned_model}")
    ft_model, ft_tokenizer = load_model(
        args.finetuned_model, 
        use_lora=args.use_lora, 
        base_model=args.base_model if args.use_lora else None
    )
    models["finetuned"] = {"model": ft_model, "tokenizer": ft_tokenizer}
    
    # Load vector store
    print(f"Loading vector store from {args.vector_store}")
    vector_store = load_vector_store(args.vector_store)

def generate(query, model_type, use_rag, temperature, k_docs):
    """Generate response from selected model."""
    global models, vector_store
    
    model = models[model_type]["model"]
    tokenizer = models[model_type]["tokenizer"]
    
    context = ""
    source_info = ""
    
    if use_rag:
        # Retrieve context
        docs = vector_store.similarity_search(query, k=k_docs)
        
        clean_context = []
        sources = []
        
        for doc in docs:
            content = doc.page_content
            content = re.sub(r'<<[^>]+>>', '', content)
            content = re.sub(r'<<[^>]+_END>>', '', content)
            clean_context.append(content)
            
            metadata = doc.metadata
            sources.append(f"- {metadata.get('title', 'Untitled')} (from {metadata.get('filename', 'unknown')})")
        
        context = "\n\n".join(clean_context)
        source_info = "\n".join(sources)
        
        # Create RAG prompt
        prompt = f"<s>[INST] I need information about the following topic. Use the provided context to answer my question accurately.\n\nContext:\n{context}\n\nQuestion: {query} [/INST]"
    else:
        # Direct prompt
        prompt = f"<s>[INST] {query} [/INST]"
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=2048,
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
    
    return response, context, source_info

def compare_models(query, use_rag_base, use_rag_ft, temp_base, temp_ft, k_docs):
    """Compare base and fine-tuned models."""
    # Generate with base model
    base_response, base_context, base_sources = generate(
        query, "base", use_rag_base, temp_base, k_docs
    )
    
    # Generate with fine-tuned model
    ft_response, ft_context, ft_sources = generate(
        query, "finetuned", use_rag_ft, temp_ft, k_docs
    )
    
    return base_response, base_context, base_sources, ft_response, ft_context, ft_sources

def create_ui():
    """Create Gradio UI for model comparison."""
    with gr.Blocks(title="Model Comparison") as demo:
        gr.Markdown("# Model Comparison Tool")
        gr.Markdown("Compare base model and fine-tuned model responses")
        
        with gr.Row():
            query = gr.Textbox(label="Your Question", lines=3)
            k_docs = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Number of Documents")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base Model")
                use_rag_base = gr.Checkbox(label="Use RAG", value=True)
                temp_base = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature")
                base_response = gr.Textbox(label="Response", lines=10)
                base_context = gr.Textbox(label="Context Used", lines=6, visible=False)
                base_sources = gr.Textbox(label="Sources", lines=3)
            
            with gr.Column():
                gr.Markdown("### Fine-tuned Model")
                use_rag_ft = gr.Checkbox(label="Use RAG", value=True)
                temp_ft = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature")
                ft_response = gr.Textbox(label="Response", lines=10)
                ft_context = gr.Textbox(label="Context Used", lines=6, visible=False)
                ft_sources = gr.Textbox(label="Sources", lines=3)
        
        # Toggle context visibility
        show_context = gr.Checkbox(label="Show Context", value=False)
        show_context.change(
            lambda x: [gr.update(visible=x), gr.update(visible=x)],
            inputs=[show_context],
            outputs=[base_context, ft_context]
        )
        
        # Submit button
        submit_btn = gr.Button("Compare Models")
        submit_btn.click(
            fn=compare_models,
            inputs=[query, use_rag_base, use_rag_ft, temp_base, temp_ft, k_docs],
            outputs=[base_response, base_context, base_sources, ft_response, ft_context, ft_sources]
        )
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Comparison UI")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", 
                      help="Base model name or path")
    parser.add_argument("--finetuned_model", type=str, required=True,
                      help="Fine-tuned model name or path")
    parser.add_argument("--use_lora", action="store_true",
                      help="Whether fine-tuned model uses LoRA")
    parser.add_argument("--vector_store", type=str, required=True,
                      help="Path to vector store")
    parser.add_argument("--port", type=int, default=7860,
                      help="Port for Gradio server")
    
    args = parser.parse_args()
    
    # Setup models
    setup_models(args)
    
    # Create and launch UI
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port)
