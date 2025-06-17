from typing import List, Dict
import numpy as np
from time import time

class RAGEvaluator:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
    def evaluate_retrieval(
        self,
        eval_data: List[Dict],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """
        Evaluate retrieval performance using various metrics.
        
        Returns:
            Dict containing:
            - recall@k: proportion of relevant passages found in top k results
            - mrr: Mean Reciprocal Rank
            - latency: average query time in seconds
        """
        recalls = {k: [] for k in k_values}
        mrr_scores = []
        latencies = []
        
        for item in eval_data:
            # Time the search
            start_time = time()
            results = self.vector_store.search(item["question"], limit=max(k_values))
            latency = time() - start_time
            latencies.append(latency)
            
            # Get retrieved IDs
            retrieved_ids = [r["id"] for r in results]
            relevant_ids = item["relevant_passage_ids"]
            
            # Calculate recall@k
            for k in k_values:
                retrieved_at_k = set(retrieved_ids[:k])
                relevant = set(relevant_ids)
                recall = len(retrieved_at_k.intersection(relevant)) / len(relevant)
                recalls[k].append(recall)
            
            # Calculate MRR
            mrr = 0
            for rank, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_ids:
                    mrr = 1.0 / rank
                    break
            mrr_scores.append(mrr)
        
        # Aggregate metrics
        metrics = {
            f"recall@{k}": np.mean(recalls[k]) for k in k_values
        }
        metrics.update({
            "mrr": np.mean(mrr_scores),
            "mean_latency": np.mean(latencies),
            "p90_latency": np.percentile(latencies, 90)
        })
        
        return metrics 