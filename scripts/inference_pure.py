import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import re


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

def generate_response(model, tokenizer, prompt, max_length=2048, temperature=0.7, top_p=0.9):
    """Generate a response from the model."""
    # Format the prompt according to the Mistral chat template
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    

    print(f"DEBUG: Prompt: {formatted_prompt[:100]}...")

    print("starting tokenizer....\n")

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
    

    print(f"DEBUG: Raw output: {response}")


    # Extract only the model's response (after [/INST])

    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()

    
    # Remove the final </s> if present
    if response.endswith("</s>"):
        response = response[:-4].strip()
    

    response = re.sub(r'\[ANS\]|\[/AN.*?$', '', response).strip()


    return response

def run_interactive_console(model, tokenizer):
    """Run an interactive console for chatting with the model."""
    print("Interactive console starting. Type 'exit' to quit.")
    print("---------------------------------------------------")
    
    while True:
        # Get user input
        user_input = input("\nUser: ")
        
        # Check if the user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Generate a response
        print("\nGenerating response...")
        response = generate_response(model, tokenizer, user_input)
        
        # Print the response
        print(f"\nModel: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--device", type=str, default="auto", help="Device to run inference on")
    parser.add_argument("--use_lora", action="store_true", help="Whether the model is a LoRA model")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name if using LoRA")
    
    args = parser.parse_args()
    
    # Load the model
    model, tokenizer = load_model(args.model_path, args.device, args.use_lora, args.base_model)
    
    # Run the interactive console
    run_interactive_console(model, tokenizer)
