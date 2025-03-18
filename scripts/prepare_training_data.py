import json
import os
import argparse
import random
from tqdm import tqdm

def create_instruction_dataset(jsonl_file, output_file, num_examples_per_doc=5):
    """Convert processed documents into an instruction dataset for fine-tuning."""
    all_examples = []
    
    # Read JSONL file
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        documents = [json.loads(line) for line in f]
    
    print(f"Generating instruction dataset from {len(documents)} documents...")
    
    for doc in tqdm(documents):
        content = doc['content']
        metadata = doc['metadata']
        title = metadata.get('title', 'Untitled Document')
        
        # Generate synthetic instruction-response pairs
        examples = generate_examples(content, title, num_examples=num_examples_per_doc)
        all_examples.extend(examples)
    
    # Shuffle examples
    random.shuffle(all_examples)
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Created {len(all_examples)} instruction-response pairs in {output_file}")
    return all_examples

def generate_examples(content, title, num_examples=5):
    """Generate synthetic instruction-response pairs from document content."""
    examples = []
    
    # Split content into paragraphs
    paragraphs = [p for p in content.split('\n\n') if p.strip()]
    
    # If there are fewer paragraphs than num_examples, create fewer examples
    num_examples = min(num_examples, len(paragraphs))
    
    # Randomly select paragraphs to create examples from
    selected_paragraphs = random.sample(paragraphs, num_examples) if len(paragraphs) >= num_examples else paragraphs
    
    for paragraph in selected_paragraphs:
        # Create different types of instructions
        instruction_types = [
            f"What does the document '{title}' say about this topic? {get_topic_from_paragraph(paragraph)}",
            f"Summarize the following information from the document '{title}': {paragraph}",
            f"Extract key points from this passage in the document '{title}': {paragraph}",
            f"Based on the document '{title}', explain the following concept: {get_topic_from_paragraph(paragraph)}",
            f"I'm reading the document '{title}'. Can you help me understand this part? {paragraph}"
        ]
        
        # Select a random instruction type
        instruction = random.choice(instruction_types)
        
        # The response is a formatted version of the paragraph
        response = format_response(paragraph, title)
        
        examples.append({
            "instruction": instruction,
            "response": response
        })
    
    return examples

def get_topic_from_paragraph(paragraph):
    """Extract a potential topic from a paragraph by taking the first sentence."""
    # Simple heuristic - take the first 10 words
    words = paragraph.split()[:10]
    return ' '.join(words) + "..."

def format_response(paragraph, title):
    """Format a response based on the paragraph content."""
    response = f"According to the document '{title}', {paragraph}"
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create instruction dataset for fine-tuning")
    parser.add_argument("--jsonl_file", type=str, required=True, help="Path to processed JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output instruction dataset")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of examples to generate per document")
    
    args = parser.parse_args()
    create_instruction_dataset(args.jsonl_file, args.output_file, args.num_examples)
