# utils/context_analysis.py
import re
from typing import List, Dict, Any, Tuple

def analyze_retrieved_documents(docs: List[Any], query: str) -> List[Tuple[Any, float]]:
    """
    Analyze retrieved documents to score their relevance to the query.
    
    Args:
        docs: List of document objects with page_content and metadata
        query: The user query string
        
    Returns:
        List of tuples (document, relevance_score)
    """
    query_words = set(query.lower().split())
    scored_docs = []
    
    for doc in docs:
        content = doc.page_content.lower()
        
        # Calculate a simple relevance score
        word_matches = sum(1 for word in query_words if word in content)
        query_coverage = word_matches / len(query_words) if query_words else 0
        
        # Check for exact phrase matches
        exact_matches = 0
        for i in range(3, len(query)):  # Check phrases of length 3+
            for j in range(len(query) - i + 1):
                phrase = query[j:j+i].lower()
                if len(phrase.split()) > 1 and phrase in content:  # Only multi-word phrases
                    exact_matches += 1
        
        # Calculate final score (weight exact matches more heavily)
        score = query_coverage * 0.7 + (0.3 * min(exact_matches, 3) / 3) 
        
        scored_docs.append((doc, score))
    
    # Sort by score, descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs

def rerank_documents(docs: List[Any], query: str, top_k: int = None) -> List[Any]:
    """Rerank documents based on relevance to query and return top k."""
    scored_docs = analyze_retrieved_documents(docs, query)
    
    # Take all documents if top_k is None, otherwise take top k
    reranked_docs = [doc for doc, _ in scored_docs[:top_k if top_k else len(scored_docs)]]
    return reranked_docs
