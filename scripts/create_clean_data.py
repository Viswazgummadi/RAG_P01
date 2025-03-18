import os
import json
import re
import random
import argparse
from tqdm import tqdm

def clean_text(text):
    """Clean special tags from text."""
    # Remove special tags
    text = re.sub(r'<<[^>]+>>', '', text)
    text = re.sub(r'<<[^>]+_END>>', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def create_instruction_dataset(input_file, output_file, num_examples_per_doc=5):
    """Create a clean instruction dataset from the processed documents."""
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        documents = [json.loads(line) for line in f]
    
    print(f"Creating instruction dataset from {len(documents)} documents")
    
    all_examples = []
    
    for doc in tqdm(documents):
        content = doc['content']
        metadata = doc['metadata']
        title = metadata.get('title', 'Untitled Document')
        
        # Clean the content
        clean_content = clean_text(content)
        
        # Split into paragraphs
        paragraphs = [p for p in clean_content.split('\n\n') if len(p.strip()) > 50]
        
        if not paragraphs:
            continue
        
        # Create examples
        num_to_create = min(num_examples_per_doc, len(paragraphs))
        selected_paragraphs = random.sample(paragraphs, num_to_create) if len(paragraphs) > num_to_create else paragraphs
        
        for para in selected_paragraphs:
            # Create question
            topic_words = para.split()[:5]
            topic = " ".join(topic_words) + "..."
            
            # Create different question types
            questions = [
                f"What does the document say about {topic}?",
                f"Explain the importance of proper cocoa bean storage.",
                f"What are the factors that can damage cocoa beans during storage?",
                f"What requirements does the standard mention for cocoa bean storage structures?",
                f"Why is it important to store cocoa beans properly?"
            ]
            
            question = random.choice(questions)
            
            # Create answer - use the whole paragraph as the basis
            answer = f"According to the document, {para}"
            
            # Create example
            example = {
                "instruction": question,
                "response": answer
            }
            
            all_examples.append(example)
    
    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Created {len(all_examples)} examples in {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create clean instruction dataset")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of examples per document")
    
    args = parser.parse_args()
    create_instruction_dataset(args.input_file, args.output_file, args.num_examples)
