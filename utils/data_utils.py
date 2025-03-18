import json
import os
import random
from typing import List, Dict, Any, Optional, Union, Tuple
import torch
from tqdm import tqdm

def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Write a list of dictionaries to a JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def split_dataset(data: List[Dict[str, Any]], train_ratio: float = 0.8, 
                  val_ratio: float = 0.1, test_ratio: float = 0.1, 
                  seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a dataset into train, validation, and test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Shuffle the data
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split indices
    n = len(shuffled_data)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    
    # Split the data
    train_data = shuffled_data[:train_end]
    val_data = shuffled_data[train_end:val_end]
    test_data = shuffled_data[val_end:]
    
    return train_data, val_data, test_data

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # If we're not at the end, try to find a good splitting point
        if end < len(text):
            # Look for a newline or period to split on
            split_point = text.rfind('\n', start, end)
            if split_point == -1 or split_point <= start:
                split_point = text.rfind('. ', start, end)
                if split_point == -1 or split_point <= start:
                    split_point = text.rfind(' ', start, end)
                    if split_point == -1 or split_point <= start:
                        split_point = end
                    else:
                        split_point += 1  # Include the space
                else:
                    split_point += 2  # Include the period and space
            else:
                split_point += 1  # Include the newline
            
            chunk = text[start:split_point]
            chunks.append(chunk)
            start = split_point - chunk_overlap
        else:
            # If we're at the end, just add the remaining text
            chunks.append(text[start:])
            break
    
    return chunks

def create_instruction_dataset(documents: List[Dict[str, Any]], num_examples_per_doc: int = 5, 
                               seed: int = 42) -> List[Dict[str, str]]:
    """Create instruction-response pairs from documents."""
    random.seed(seed)
    all_examples = []
    
    for doc in tqdm(documents, desc="Generating instruction dataset"):
        content = doc['content']
        metadata = doc['metadata']
        title = metadata.get('title', 'Untitled Document')
        
        # Split content into paragraphs
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        
        # If there are fewer paragraphs than num_examples, create fewer examples
        num_examples = min(num_examples_per_doc, len(paragraphs))
        
        # Randomly select paragraphs to create examples from
        selected_paragraphs = random.sample(paragraphs, num_examples) if len(paragraphs) >= num_examples else paragraphs
        
        for paragraph in selected_paragraphs:
            # Extract a potential topic from the paragraph
            words = paragraph.split()[:10]
            topic = ' '.join(words) + "..."
            
            # Create different types of instructions
            instruction_types = [
                f"What does the document '{title}' say about this topic? {topic}",
                f"Summarize the following information from the document '{title}': {paragraph[:100]}...",
                f"Extract key points from this passage in the document '{title}': {paragraph[:100]}...",
                f"Based on the document '{title}', explain the following concept: {topic}",
                f"I'm reading the document '{title}'. Can you help me understand this part? {paragraph[:100]}..."
            ]
            
            # Select a random instruction type
            instruction = random.choice(instruction_types)
            
            # The response is a formatted version of the paragraph
            response = f"According to the document '{title}', {paragraph}"
            
            all_examples.append({
                "instruction": instruction,
                "response": response
            })
    
    # Shuffle examples
    random.shuffle(all_examples)
    
    return all_examples

def format_mistral_prompt(instruction: str, response: Optional[str] = None) -> str:
    """Format a prompt according to the Mistral chat template."""
    if response:
        return f"<s>[INST] {instruction} [/INST] {response} </s>"
    else:
        return f"<s>[INST] {instruction} [/INST]"
