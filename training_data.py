import os
import json
import re
import random
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

def create_training_data(jsonl_file, output_dir, samples_per_doc=5):
    """Create training data for fine-tuning the Mistral model."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Question templates for training data generation
    question_templates = [
        "What are the main points discussed in {title}?",
        "Summarize the key information in the document about {subject}.",
        "What does {title} say about {subject}?",
        "Extract the important facts about {subject} from {title}.",
        "What are the conclusions presented in {title} regarding {subject}?",
        "Explain the concept of {subject} as described in the document.",
        "What evidence supports {subject} according to {title}?",
        "How does {title} define {subject}?",
        "What are the requirements for {subject} mentioned in {title}?",
        "What is the relationship between {subject} and other concepts in {title}?"
    ]
    
    training_data = []
    
    # Load documents
    print("Loading documents...")
    documents = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            documents.append(json.loads(line))
    
    print(f"Generating training data from {len(documents)} documents")
    
    for doc in tqdm(documents, desc="Creating training samples"):
        content = doc['content']
        metadata = doc['metadata']
        title = metadata.get('title', 'the document')
        
        # Extract potential subjects from headings and content
        heading_pattern = r'(#+)\s+(.*)'
        headings = re.findall(heading_pattern, content)
        
        potential_subjects = []
        
        # From headings
        for _, heading_text in headings:
            words = heading_text.strip().split()
            for word in words:
                if len(word) > 5 and word[0].isupper():
                    potential_subjects.append(word)
        
        # From content
        words = content.split()
        for word in words:
            if len(word) > 5 and word[0].isupper() and word not in potential_subjects:
                potential_subjects.append(word)
        
        # Limit subjects and ensure we have some
        potential_subjects = list(set(potential_subjects))[:20]
        if not potential_subjects:
            potential_subjects = ["this topic", "the main subject", "the key concept"]
        
        # Generate training samples for this document
        for _ in range(samples_per_doc):
            # Select a random question template and subject
            template = random.choice(question_templates)
            subject = random.choice(potential_subjects)
            
            # Create the question
            question = template.format(title=title, subject=subject)
            
            # Create instruction format following Mistral's format
            instruction = f"""Answer the following question based only on the provided document content.
Include the source document title and filename in your answer.

Question: {question}

Document Content: {content[:3000]}..."""
            
            # Create a synthetic answer with source attribution
            response = f"""Based on the document '{title}', {subject} is discussed in detail. The document provides information about its definition, characteristics, and importance. The document explains that {subject} is a key concept within the context of {title}.

Source: {metadata['title']} ({metadata['filename']})"""
            
            # Add to training data
            training_data.append({
                "instruction": instruction,
                "output": response
            })
    
    # Convert to Hugging Face Dataset
    df = pd.DataFrame(training_data)
    dataset = Dataset.from_pandas(df)
    
    # Save the dataset
    dataset.save_to_disk(output_dir)
    print(f"Training dataset with {len(dataset)} samples saved to {output_dir}")
    
    return dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create training data for fine-tuning")
    parser.add_argument("--jsonl_file", type=str, required=True, help="Path to processed JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save training dataset")
    parser.add_argument("--samples_per_doc", type=int, default=5, help="Number of samples per document")
    
    args = parser.parse_args()
    create_training_data(args.jsonl_file, args.output_dir, args.samples_per_doc)
