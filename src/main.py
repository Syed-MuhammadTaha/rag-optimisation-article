from data_loader import BioASQDataLoader
from vector_store import VectorStore
from generator import Generator
from evaluator import RAGEvaluator
import json
from pathlib import Path

def main():
    # Initialize components
    data_loader = BioASQDataLoader()
    vector_store = VectorStore()
    generator = Generator()
    evaluator = RAGEvaluator(vector_store)
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Load data
    print("Loading corpus...")
    corpus = data_loader.load_corpus()
    eval_data = data_loader.load_eval_data()
    
    # Setup vector store
    print("Creating collection...")
    vector_store.create_collection()
    
    print("Inserting passages...")
    vector_store.insert_passages(corpus)
    
    # Example end-to-end pipeline
    print("\nTesting end-to-end pipeline...")
    example_question = eval_data[0]["question"]
    print(f"Question: {example_question}")
    
    # Retrieve relevant passages
    retrieved_passages = vector_store.search(example_question, limit=3)
    print(f"\nRetrieved {len(retrieved_passages)} passages")
    
    # Generate answer
    answer = generator.generate(example_question, retrieved_passages)
    print(f"\nGenerated Answer: {answer}")
    print(f"Ground Truth: {eval_data[0]['answer']}")
    
    # Run evaluation
    print("\nEvaluating retrieval performance...")
    metrics = evaluator.evaluate_retrieval(eval_data)
    
    # Save results
    print("\nRetrieval Metrics:")
    print(json.dumps(metrics, indent=2))
    
    with open("results/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main() 