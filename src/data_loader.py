from datasets import load_dataset
from typing import Dict, List

class BioASQDataLoader:
    def __init__(self):
        self.dataset = load_dataset("enelpol/rag-mini-bioasq")
        
    def load_corpus(self) -> List[Dict]:
        """Load the text corpus passages."""
        corpus = self.dataset["text-corpus"]["train"]
        return [
            {
                "id": str(row["id"]),  # Convert to string for Qdrant
                "text": row["text"],
            }
            for row in corpus
        ]
    
    def load_eval_data(self) -> List[Dict]:
        """Load all question-answer evaluation data."""
        qa_data = self.dataset["question-answer-passages"]["train"]
        return [
            {
                "id": str(row["id"]),
                "question": row["question"],
                "answer": row["answer"],
                "relevant_passage_ids": [str(pid) for pid in row["relevant_passage_ids"]]
            }
            for row in qa_data
        ] 