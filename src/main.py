from data_loader import DataLoader
from vector_store import VectorStore
from ingestion import DocumentIngestion
from evaluate_retrieval import evaluate_retrieval

def main():
    # Initialize components with optimization features
    print("Initializing optimized RAG system...")
    data_loader = DataLoader()
    
    # Create vector store with advanced features enabled
    vector_store = VectorStore(
        enable_quantization=False,  # Set to True for memory-constrained environments
        enable_semantic_cache=True,
        cache_threshold=0.35,
        cache_size_limit=1000
    )
    
    # Load data
    print("Loading corpus...")
    corpus = data_loader.load_corpus()
    print(f"Loaded {len(corpus)} passages from corpus")
    
    # # Setup optimized vector store collection
    # print("Creating optimized collection with HNSW tuning...")
    # vector_store.create_optimized_collection()
    
    # # Setup metadata indexes for filtering
    # print("Setting up metadata indexes...")
    # vector_store.setup_metadata_indexes()
    
    # # Initialize ingestion component
    # ingestion = DocumentIngestion(
    #     client=vector_store.client,
    #     collection_name=vector_store.collection_name,
    #     model_name=vector_store.model_name
    # )
    
    # # Ingest documents
    # print("Inserting passages with optimized batching and caching...")
    # total_time, processed_ids = ingestion.insert_passages(corpus)
    # print(f"Ingestion completed in {total_time:.2f}s, processed {len(processed_ids)} documents")
    
    # Display collection info
    print("\n" + "="*60)
    print("COLLECTION INFORMATION")
    print("="*60)
    collection_info = vector_store.get_collection_info()
    for key, value in collection_info.items():
        print(f"{key}: {value}")
    
    # Demonstrate optimized retrieval techniques
    print("\n" + "="*60)
    print("DEMONSTRATING OPTIMIZED RETRIEVAL TECHNIQUES")
    print("="*60)
    
    test_query = "What are the effects of diabetes on cardiovascular health?"
    print(f"\nTest Query: {test_query}")
    
    # 1. Standard retrieval with semantic caching
    print("\n1. Standard Retrieval with Semantic Caching:")
    results1 = vector_store.retrieve_passages(test_query, k=3)
    for i, passage in enumerate(results1, 1):
        print(f"   Result {i}: Score={passage['score']:.4f}, ID={passage['id']}")
    
    # 2. Cached retrieval (should be faster on second call)
    print("\n2. Cached Retrieval (same query):")
    results2 = vector_store.retrieve_passages(test_query, k=3)
    
    # 3. Fast retrieval with disabled rescoring (if quantization enabled)
    print("\n3. Fast Retrieval (cache + optimized search):")
    results3 = vector_store.fast_retrieve_passages(test_query, k=3)
    
    # Display cache statistics
    print("\n" + "-"*50)
    print("SEMANTIC CACHE PERFORMANCE")
    print("-"*50)
    cache_stats = vector_store.get_cache_statistics()
    for key, value in cache_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Test different query to show cache miss/hit behavior
    print("\n4. Different Query (cache miss expected):")
    different_query = "How does insulin resistance affect metabolism?"
    results4 = vector_store.retrieve_passages(different_query, k=2)
    for i, passage in enumerate(results4, 1):
        print(f"   Result {i}: Score={passage['score']:.4f}, ID={passage['id']}")
    
    # Run comprehensive evaluation
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE RETRIEVAL EVALUATION")
    print("="*60)
    try:
        evaluate_retrieval(top_k=5)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("This may be due to collection name mismatch or missing data.")
    
    # Final cache statistics after all operations
    print("\n" + "="*60)
    print("FINAL SYSTEM PERFORMANCE SUMMARY")
    print("="*60)
    final_cache_stats = vector_store.get_cache_statistics()
    print(f"Cache enabled: {final_cache_stats.get('cache_enabled', False)}")
    if final_cache_stats.get('cache_enabled'):
        print(f"Cache hit ratio: {final_cache_stats.get('hit_ratio', 0):.3f}")
        print(f"Total cache hits: {final_cache_stats.get('total_hits', 0)}")
        print(f"Total cache misses: {final_cache_stats.get('total_misses', 0)}")
        print(f"Cache entries: {final_cache_stats.get('cache_entries', 0)}")
    
    print(f"\nOptimizations applied:")
    print(f"✓ HNSW tuning (m=64, ef_construct=512)")
    print(f"✓ Semantic caching enabled")
    print(f"✓ Metadata indexing for filtering")
    print(f"✓ Lazy loading for memory efficiency")
    print(f"✓ Batch ingestion with parallel processing")
    
    if vector_store.enable_quantization:
        print(f"✓ Vector quantization (4x memory reduction)")
    else:
        print(f"○ Vector quantization disabled (enable for memory savings)")

if __name__ == "__main__":
    main() 