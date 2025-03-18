import torch
import os
from typing import Dict, Any, Optional, Union, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig,
    prepare_model_for_kbit_training
)

def load_model_and_tokenizer(
    model_name_or_path: str,
    use_4bit: bool = True,
    use_lora: bool = False,
    lora_config: Optional[Dict[str, Any]] = None,
    device: str = "auto",
    bf16: bool = True,
    is_adapter_path: bool = False,
    base_model: Optional[str] = None
) -> Tuple[Union[AutoModelForCausalLM, PeftModel], AutoTokenizer]:
    """
    Load a language model and its tokenizer with various configurations.
    
    Args:
        model_name_or_path: Name or path of the model to load
        use_4bit: Whether to quantize the model to 4-bit precision
        use_lora: Whether to use LoRA
        lora_config: Configuration for LoRA
        device: Device to load the model on
        bf16: Whether to use bfloat16 precision
        is_adapter_path: Whether model_name_or_path is a LoRA adapter path
        base_model: Base model path for LoRA
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Handle LoRA adapter path
    if is_adapter_path:
        if base_model is None:
            config = PeftConfig.from_pretrained(model_name_or_path)
            base_model = config.base_model_name_or_path
        
        # Load base model
        model = load_base_model(base_model, use_4bit, device, bf16)
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        # Load model normally
        model = load_base_model(model_name_or_path, use_4bit, device, bf16)
        
        # Apply LoRA if needed
        if use_lora:
            model = prepare_model_for_kbit_training(model)
            if lora_config is None:
                lora_config = {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
                }
            
            peft_config = LoraConfig(**lora_config)
            model = get_peft_model(model, peft_config)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Fix tokenizer pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_base_model(
    model_name_or_path: str,
    use_4bit: bool = True,
    device: str = "auto",
    bf16: bool = True
) -> AutoModelForCausalLM:
    """Load a base model with optional quantization."""
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float32,
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map=device,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if bf16 else torch.float32,
            device_map=device,
        )
    
    return model

def create_training_args(
    output_dir: str,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    num_train_epochs: int = 3,
    bf16: bool = True,
    logging_steps: int = 10,
    save_strategy: str = "epoch",
    save_total_limit: int = 3,
    local_rank: int = 0
) -> TrainingArguments:
    """Create training arguments for Hugging Face Trainer."""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        bf16=bf16,
        local_rank=local_rank,
    )

def generate_text(
    model: Union[AutoModelForCausalLM, PeftModel],
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
) -> str:
    """Generate text from a prompt using the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the model's response (after [/INST])
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    
    # Remove the final </s> if present
    if response.endswith("</s>"):
        response = response[:-4].strip()
    
    return response

def save_model_and_tokenizer(
    model: Union[AutoModelForCausalLM, PeftModel],
    tokenizer: AutoTokenizer,
    output_dir: str
) -> None:
    """Save model and tokenizer to disk."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")
