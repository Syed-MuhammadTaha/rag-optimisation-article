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
            result.append({
                "id": str(idx),
                "question": row["question"],
                "answer": row["answer"],
                "relevant_passage_ids": [str(pid) for pid in row["relevant_passage_ids"]]
            })
        return result

    def filter_qa_dataset(self, qa_pairs: List[Dict], valid_passage_ids: List[str]) -> List[Dict]:
        """
        Filter QA pairs to only include those whose relevant passages were ingested.
        
        Args:
            qa_pairs: List of QA pairs with relevant passage IDs
            valid_passage_ids: List of passage IDs that were successfully ingested
            
        Returns:
            List of filtered QA pairs
        """
        valid_passage_ids = set(valid_passage_ids)  # Convert to set for O(1) lookup
        
        filtered_qa_pairs = []
        for qa_pair in qa_pairs:
            # Assuming each qa_pair has a 'relevant_passage_ids' field
            # Modify this based on your actual data structure
            if isinstance(qa_pair.get('relevant_passage_ids'), list):
                # Only keep QA pairs where ALL relevant passages were ingested
                if all(pid in valid_passage_ids for pid in qa_pair['relevant_passage_ids']):
                    filtered_qa_pairs.append(qa_pair)
            elif isinstance(qa_pair.get('relevant_passage_id'), str):
                # If there's only one relevant passage ID
                if qa_pair['relevant_passage_id'] in valid_passage_ids:
                    filtered_qa_pairs.append(qa_pair)
        
        print(f"Filtered QA pairs from {len(qa_pairs)} to {len(filtered_qa_pairs)}")
        return filtered_qa_pairs