from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_from_disk

def finetune_mistral(dataset_path, output_dir):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_from_disk(dataset_path).train_test_split(test_size=0.1)
    
    def tokenize_function(examples):
        prompts = [f"<s>[INST] {instruction} [/INST] {output}</s>" for instruction, output in zip(examples['instruction'], examples['output'])]
        return tokenizer(prompts, truncation=True, max_length=2048)

    tokenized_train_dataset = dataset['train'].map(tokenize_function)
    tokenized_val_dataset = dataset['test'].map(tokenize_function)

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model.gradient_checkpointing_enable()
    
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj"], lora_dropout=0.05)
    model = get_peft_model(model.prepare_model_for_kbit_training(), lora_config)

    training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=3,
                                       per_device_train_batch_size=4,
                                       save_steps=100)

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=tokenized_train_dataset,
                      eval_dataset=tokenized_val_dataset)

    trainer.train()
