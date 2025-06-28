from data_loader import DataLoader
from vector_store import VectorStore
from ingestion import DocumentIngestion

def main():
    # Initialize components
    data_loader = DataLoader()
    vector_store = VectorStore()
    
    # Load data
    print("Loading corpus...")
    corpus = data_loader.load_corpus()
    
    # Setup vector store collection
    print("Creating collection...")
    vector_store.create_collection()
    
    # Initialize ingestion component
    ingestion = DocumentIngestion(
        client=vector_store.client,
        collection_name=vector_store.collection_name,
        model_name=vector_store.model_name
    )
    
    # Ingest documents
    print("Inserting passages...")
    total_time, processed_ids = ingestion.insert_passages(corpus)
    print(f"Ingestion completed in {total_time:.2f}s, processed {len(processed_ids)} documents")
    
    # Display collection info
    print("\n" + "="*50)
    print("COLLECTION INFORMATION")
    print("="*50)
    collection_info = vector_store.get_collection_info()
    for key, value in collection_info.items():
        print(f"{key}: {value}")
    
    # Demonstrate retrieval functionality
    print("\n" + "="*50)
    print("DEMONSTRATING LANGCHAIN RETRIEVAL")
    print("="*50)
    
    # Test query
    test_query = "What are the effects of diabetes on cardiovascular health?"
    print(f"\nQuery: {test_query}")
    
    # Retrieve similar passages
    print("\nRetrieving top 3 similar passages...")
    retrieved_passages = vector_store.retrieve_passages(test_query, k=3)
    print(retrieved_passages)
    
    # # Display results
    # for i, passage in enumerate(retrieved_passages, 1):
    #     print(f"\n--- Result {i} (Score: {passage['score']:.4f}) ---")
    #     print(f"ID: {passage['id']}")
    #     print(f"Text: {passage['text'][:200]}...")

if __name__ == "__main__":
    main() 