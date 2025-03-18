import os
import torch
import argparse
import json
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from tqdm import tqdm

def format_instruction(example):
    """Format the instruction and response into the Mistral chat template"""
    return f"<s>[INST] {example['instruction']} [/INST] {example['response']} </s>"

def prepare_training_dataset(dataset_file):
    """Load and prepare the training dataset"""
    # Load the dataset
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(data)
    
    return dataset

def tokenize_dataset(dataset, tokenizer, max_length=2048):
    """Tokenize the dataset"""
    def tokenize_function(examples):
        # Format the examples
        formatted_texts = [format_instruction({"instruction": inst, "response": resp}) 
                           for inst, resp in zip(examples["instruction"], examples["response"])]
        
        # Tokenize the texts
        tokenized = tokenizer(
            formatted_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Add labels for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Apply tokenization to the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return tokenized_dataset

def train_model(
    dataset_file,
    model_name="mistralai/Mistral-7B-v0.1",
    output_dir="models/checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    local_rank=0,
):
    """Fine-tune the model"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading and preparing dataset...")
    dataset = prepare_training_dataset(dataset_file)
    
    # Compute bnb kwargs for quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float32,
        bnb_4bit_use_double_quant=False,
    )
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Add special tokens if needed
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare the model for training with LoRA if enabled
    if use_lora:
        print("Setting up LoRA for efficient fine-tuning...")
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, peft_config)
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        bf16=bf16,
        local_rank=local_rank,
    )
    
    # Initialize the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the trained model
    print(f"Saving model to {output_dir}/final")
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Mistral model")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to instruction dataset JSONL file")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad_accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--no_lora", action="store_false", dest="use_lora", help="Disable LoRA fine-tuning")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    train_model(
        dataset_file=args.dataset_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora,
        local_rank=args.local_rank,
    )
