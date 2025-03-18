import argparse
import json
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_model(model_path, device="auto", use_lora=True, base_model=None):
    """Load model with or without LoRA."""
    if use_lora:
        if base_model is None:
            config = PeftConfig.from_pretrained(model_path)
            base_model = config.base_model_name_or_path
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        model = PeftModel.from_pretrained(model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_vector_store(vector_store_path):
    """Load the vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

def retrieve_context(vector_store, query, k=3):
    """Retrieve and clean context."""
    retrieved_docs = vector_store.similarity_search(query, k=k)
    
    clean_context = []
    for doc in retrieved_docs:
        content = doc.page_content
        content = re.sub(r'<<[^>]+>>', '', content)
        content = re.sub(r'<<[^>]+_END>>', '', content)
        clean_context.append(content)
    
    context = "\n\n".join(clean_context)
    sources = [doc.metadata for doc in retrieved_docs]
    
    return context, sources

def generate_response(model, tokenizer, query, context=None, max_length=2048, temperature=0.7):
    """Generate response from model."""
    if context:
        # RAG prompt
        prompt = f"<s>[INST] I need information about the following topic. Use the provided context to answer my question accurately.\n\nContext:\n{context}\n\nQuestion: {query} [/INST]"
    else:
        # Direct prompt
        prompt = f"<s>[INST] {query} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
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
    
    return response

def compute_metrics(prediction, reference):
    """Compute BLEU and ROUGE scores."""
    # BLEU score
    try:
        reference_tokens = nltk.word_tokenize(reference.lower())
        prediction_tokens = nltk.word_tokenize(prediction.lower())
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoothing)
    except:
        bleu = 0
    
    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    
    return {
        'bleu': bleu,
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def evaluate_test_questions(model, tokenizer, vector_store, test_questions, use_rag=True, k_docs=3):
    """Evaluate model on test questions."""
    results = []
    
    for question in tqdm(test_questions):
        query = question['question']
        reference = question.get('reference', None)
        
        if use_rag:
            context, sources = retrieve_context(vector_store, query, k=k_docs)
            response = generate_response(model, tokenizer, query, context)
            
            # Record source documents
            source_info = [{'title': src.get('title', 'Untitled'), 
                            'filename': src.get('filename', 'unknown')} 
                           for src in sources]
        else:
            context = None
            response = generate_response(model, tokenizer, query)
            source_info = []
        
        result = {
            'question': query,
            'response': response,
            'context_used': bool(context),
            'sources': source_info
        }
        
        # If reference exists, compute metrics
        if reference:
            metrics = compute_metrics(response, reference)
            result['metrics'] = metrics
        
        results.append(result)
    
    return results

def main(args):
    # Load models
    print("Loading models...")
    
    # Load fine-tuned model
    ft_model, ft_tokenizer = load_model(
        args.finetuned_model,
        use_lora=args.use_lora,
        base_model=args.base_model
    )
    
    # Load base model if specified
    if args.base_model and args.compare_with_base:
        base_model, base_tokenizer = load_model(
            args.base_model,
            use_lora=False
        )
    else:
        base_model, base_tokenizer = None, None
    
    # Load vector store
    if args.use_rag:
        vector_store = load_vector_store(args.vector_store_path)
    else:
        vector_store = None
    
    # Load test questions
    if args.test_file:
        with open(args.test_file, 'r') as f:
            test_questions = json.load(f)
    else:
        # Default test questions if no file provided
        test_questions = [
            {'question': 'Why is it important to store cocoa beans in proper structures?'},
            {'question': 'What factors can damage cocoa beans during storage?'},
            {'question': 'What are the requirements for cocoa bean storage locations?'},
            {'question': 'How should bag storage structures for cocoa beans be constructed?'},
            {'question': 'What is the purpose of the Indian Standard for cocoa beans storage?'}
        ]
    
    # Evaluate fine-tuned model
    print("\nEvaluating fine-tuned model...")
    ft_results = evaluate_test_questions(
        ft_model, ft_tokenizer, vector_store, 
        test_questions, use_rag=args.use_rag, k_docs=args.k_docs
    )
    
    # Evaluate base model if specified
    if base_model and base_tokenizer:
        print("\nEvaluating base model...")
        base_results = evaluate_test_questions(
            base_model, base_tokenizer, vector_store, 
            test_questions, use_rag=args.use_rag, k_docs=args.k_docs
        )
    else:
        base_results = None
    
    # Compile results
    results = {
        'fine_tuned_model': args.finetuned_model,
        'base_model': args.base_model if args.compare_with_base else None,
        'rag_used': args.use_rag,
        'k_docs': args.k_docs if args.use_rag else None,
        'fine_tuned_results': ft_results,
        'base_model_results': base_results
    }
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_file}")
    
    # Print summary
    print("\nSummary:")
    print("=========")
    
    ft_empty = sum(1 for r in ft_results if len(r['response']) < 20)
    print(f"Fine-tuned model empty responses: {ft_empty}/{len(ft_results)}")
    
    if base_results:
        base_empty = sum(1 for r in base_results if len(r['response']) < 20)
        print(f"Base model empty responses: {base_empty}/{len(base_results)}")
    
    # Print metrics if available
    has_metrics = any('metrics' in r for r in ft_results)
    if has_metrics:
        ft_rouge1 = sum(r['metrics']['rouge1'] for r in ft_results if 'metrics' in r) / \
                   sum(1 for r in ft_results if 'metrics' in r)
        print(f"Fine-tuned model average ROUGE-1: {ft_rouge1:.4f}")
        
        if base_results:
            base_rouge1 = sum(r['metrics']['rouge1'] for r in base_results if 'metrics' in r) / \
                         sum(1 for r in base_results if 'metrics' in r)
            print(f"Base model average ROUGE-1: {base_rouge1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model performance")
    parser.add_argument("--finetuned_model", type=str, required=True, 
                        help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default=None, 
                        help="Base model name if using LoRA")
    parser.add_argument("--use_lora", action="store_true",
                        help="Whether the fine-tuned model is a LoRA model")
    parser.add_argument("--compare_with_base", action="store_true",
                        help="Compare with base model")
    parser.add_argument("--use_rag", action="store_true",
                        help="Use RAG for evaluation")
    parser.add_argument("--vector_store_path", type=str, default="vector_db",
                        help="Path to vector store for RAG")
    parser.add_argument("--test_file", type=str, default=None,
                        help="JSON file with test questions")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                        help="Path to save evaluation results")
    parser.add_argument("--k_docs", type=int, default=3,
                        help="Number of documents to retrieve for RAG")
    
    args = parser.parse_args()
    main(args)
