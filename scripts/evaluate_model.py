import os
import argparse
import json
import torch
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import load_model_and_tokenizer
from utils.evaluation import evaluate_model, save_evaluation_results
from utils.data_utils import read_jsonl

def main(args):
    """Main function to evaluate a fine-tuned model."""
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model_path,
        use_4bit=True,
        device="auto",
        bf16=True,
        is_adapter_path=args.use_lora,
        base_model=args.base_model
    )
    
    print(f"Loading test data from {args.test_file}...")
    test_data = read_jsonl(args.test_file)
    
    # If test_size is specified, use only a subset of the test data
    if args.test_size > 0 and args.test_size < len(test_data):
        import random
        random.seed(42)
        test_data = random.sample(test_data, args.test_size)
    
    print(f"Evaluating model on {len(test_data)} test examples...")
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        max_length=args.max_length
    )
    
    print("\nEvaluation results:")
    print(f"Average BLEU: {results['avg_bleu']:.4f}")
    print(f"Average ROUGE-1: {results['avg_rouge1']:.4f}")
    print(f"Average ROUGE-2: {results['avg_rouge2']:.4f}")
    print(f"Average ROUGE-L: {results['avg_rougeL']:.4f}")
    
    # Save results to file
    save_evaluation_results(results, args.output_file)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test file (JSONL)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save evaluation results")
    parser.add_argument("--use_lora", action="store_true", help="Whether the model is a LoRA model")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name if using LoRA")
    parser.add_argument("--test_size", type=int, default=0, help="Number of test examples (0 for all)")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length for generation")
    
    args = parser.parse_args()
    main(args)
