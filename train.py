import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def finetune_mistral(dataset_path, output_dir, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
    """Fine-tune Mistral model on custom dataset using PEFT/LoRA."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)
    
    # Split into training and validation
    print("Preparing training and validation splits...")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize function with Mistral's chat format
    def tokenize_function(examples):
        prompts = []
        for instruction, response in zip(examples['instruction'], examples['output']):
            # Format using Mistral's chat template
            prompt = f"<s>[INST] {instruction} [/INST] {response}</s>"
            prompts.append(prompt)
        
        return tokenizer(
            prompts, 
            truncation=True, 
            max_length=2048, 
            padding="max_length"
        )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=train_dataset.column_names
    )
    
    tokenized_val = val_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=val_dataset.column_names
    )
    
    # Set up quantization config for efficient training
    print("Setting up model with quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepare model for training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA configuration
    print("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    print("Trainable parameters:")
    print(model.print_trainable_parameters())
    
    # Training arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        fp16=True,
        optim="paged_adamw_8bit",
        seed=42,
        push_to_hub=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Initialize Trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the model
    final_model_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"Model training completed and saved to {final_model_path}")
    
    return final_model_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Mistral model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save fine-tuned model")
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", 
                        help="Hugging Face model ID for Mistral")
    
    args = parser.parse_args()
    finetune_mistral(args.dataset_path, args.output_dir, args.model_id)
