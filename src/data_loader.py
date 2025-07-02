from typing import Dict, List
import pandas as pd

class DataLoader:
    def __init__(self):
        # Load both configurations using pandas
        self.qa_dataset = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-bioasq/data/test.parquet/part.0.parquet")
        self.corpus_dataset = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-bioasq/data/passages.parquet/part.0.parquet")
        
    def load_corpus(self) -> List[Dict]:
        """Load the text corpus passages."""
        result = []
        for idx, row in self.corpus_dataset.iterrows():
            result.append({
                "id": str(idx),
                "text": row["passage"]
            })
        return result
    
    def load_eval_data(self) -> List[Dict]:
        """Load all question-answer evaluation data."""
        result = []
        for idx, row in self.qa_dataset.iterrows():
            # Parse relevant passage IDs - they might be a string representation of a list
            relevant_ids = row["relevant_passage_ids"]
            if isinstance(relevant_ids, str):
                # Remove brackets and split by commas
                relevant_ids = relevant_ids.strip('[]').split(',')
                # Clean up each ID
                relevant_ids = [pid.strip() for pid in relevant_ids if pid.strip()]
            elif isinstance(relevant_ids, list):
                relevant_ids = [str(pid) for pid in relevant_ids]
            else:
                relevant_ids = []
            
            result.append({
                "id": str(idx),
                "question": row["question"],
                "answer": row["answer"],
                "relevant_passage_ids": relevant_ids
            })
        return result
