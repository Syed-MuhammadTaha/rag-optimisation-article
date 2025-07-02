from data_loader import DataLoader
from vector_store import VectorStore
from typing import List, Dict, Set
import json

def calculate_metrics(retrieved_ids: Set[str], relevant_ids: Set[str]) -> Dict:
    """Calculate retrieval metrics."""
    true_positives = len(retrieved_ids.intersection(relevant_ids))
    false_positives = len(retrieved_ids - relevant_ids)
    false_negatives = len(relevant_ids - retrieved_ids)
    
    precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
    recall = true_positives / len(relevant_ids) if relevant_ids else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "retrieved_count": len(retrieved_ids),
        "relevant_count": len(relevant_ids)
    }

def evaluate_retrieval(top_k: int = 5):
    """Evaluate retrieval performance using the evaluation dataset with optimized vector store."""
    print("\n" + "="*50)
    print("RETRIEVAL EVALUATION")
    print("="*50)
    
    # Load evaluation data
    data_loader = DataLoader()
    eval_data = data_loader.load_eval_data()
    print(f"\nLoaded {len(eval_data)} evaluation questions")
    
    # Initialize optimized vector store (using existing collection)
    vector_store = VectorStore(
        enable_quantization=False,
        enable_semantic_cache=False,  # Disable cache for evaluation to get true performance
        cache_threshold=0.35,
        cache_size_limit=1000
    )
    
    # Check if collection exists
    if not vector_store.collection_exists():
        print("❌ Collection does not exist. Please run ingestion first.")
        return
    
    # Take first question for testing
    test_case = eval_data[0]
    question = test_case["question"]
    relevant_passage_ids = set(test_case["relevant_passage_ids"])
    
    print(f"\nTest Question: {question}")
    print(f"Expected Relevant Passages: {len(relevant_passage_ids)}")
    print(f"Relevant Passage IDs: {relevant_passage_ids}")
    
    # Retrieve passages using optimized retrieval (without cache for evaluation)
    print(f"\nRetrieving top {top_k} passages using optimized vector store...")
    retrieved_passages = vector_store.retrieve_passages(
        query=question, 
        k=top_k,
        use_cache=False  # Disable cache for fair evaluation
    )
    
    # Extract retrieved IDs
    retrieved_ids = set()
    for passage in retrieved_passages:
        retrieved_ids.add(passage["id"])
    
    print(f"\nRetrieved Passage IDs: {retrieved_ids}")
    
    # Calculate metrics
    metrics = calculate_metrics(retrieved_ids, relevant_passage_ids)
    
    # Print results
    print("\n" + "-"*50)
    print("EVALUATION RESULTS")
    print("-"*50)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"\nTrue Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    
    # Print detailed results
    print("\n" + "-"*50)
    print("RETRIEVED PASSAGES")
    print("-"*50)
    for i, passage in enumerate(retrieved_passages, 1):
        is_relevant = passage["id"] in relevant_passage_ids
        relevance_mark = "✓" if is_relevant else "✗"
        print(f"\n[{relevance_mark}] Result {i} (Score: {passage['score']:.4f})")
        print(f"Passage ID: {passage['id']}")
        print(f"Text Preview: {passage['text'][:200]}...")
    
    # Additional evaluation with different top_k values
    print("\n" + "-"*50)
    print("PERFORMANCE AT DIFFERENT K VALUES")
    print("-"*50)
    for k in [1, 3, 5, 10]:
        if k <= len(retrieved_ids):
            # Use first k results
            k_retrieved_ids = set(list(retrieved_ids)[:k])
        else:
            # Retrieve more if needed
            k_results = vector_store.retrieve_passages(
                query=question, 
                k=k,
                use_cache=False
            )
            k_retrieved_ids = set(passage["id"] for passage in k_results)
        
        k_metrics = calculate_metrics(k_retrieved_ids, relevant_passage_ids)
        print(f"k={k:2d}: Precision={k_metrics['precision']:.3f}, "
              f"Recall={k_metrics['recall']:.3f}, F1={k_metrics['f1']:.3f}")
    
    return metrics

if __name__ == "__main__":
    evaluate_retrieval(top_k=5) 