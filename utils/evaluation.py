import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU score between reference and hypothesis."""
    # Tokenize the strings
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    
    # Compute BLEU score
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing)

def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute ROUGE scores between reference and hypothesis."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def evaluate_model(
    model: Union[AutoModelForCausalLM, PeftModel],
    tokenizer: AutoTokenizer,
    test_data: List[Dict[str, str]],
    max_length: int = 2048
) -> Dict[str, Any]:
    """Evaluate model on test data using BLEU and ROUGE metrics."""
    results = []
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for item in tqdm(test_data, desc="Evaluating model"):
        instruction = item['instruction']
        reference = item['response']
        
        # Format the prompt according to the Mistral chat template
        formatted_prompt = f"<s>[INST] {instruction} [/INST]"
        
        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate a response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
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
        
        # Compute metrics
        bleu = compute_bleu(reference, response)
        rouge = compute_rouge(reference, response)
        
        # Store results
        results.append({
            'instruction': instruction,
            'reference': reference,
            'prediction': response,
            'bleu': bleu,
            'rouge1': rouge['rouge1'],
            'rouge2': rouge['rouge2'],
            'rougeL': rouge['rougeL']
        })
        
        bleu_scores.append(bleu)
        rouge1_scores.append(rouge['rouge1'])
        rouge2_scores.append(rouge['rouge2'])
        rougeL_scores.append(rouge['rougeL'])
    
    # Compute average scores
    avg_bleu = np.mean(bleu_scores)
    avg_rouge1 = np.mean(rouge1_scores)
    avg_rouge2 = np.mean(rouge2_scores)
    avg_rougeL = np.mean(rougeL_scores)
    
    # Return evaluation results
    return {
        'examples': results,
        'avg_bleu': float(avg_bleu),
        'avg_rouge1': float(avg_rouge1),
        'avg_rouge2': float(avg_rouge2),
        'avg_rougeL': float(avg_rougeL)
    }

def save_evaluation_results(results: Dict[str, Any], output_file: str) -> None:
    """Save evaluation results to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to {output_file}")
