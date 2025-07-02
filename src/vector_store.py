from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import hashlib
import time
import uuid
from datetime import datetime, timedelta


class VectorStore:
    """
    Optimized Qdrant vector store with advanced retrieval techniques:
    - HNSW parameter tuning
    - Vector quantization and rescoring
    - Metadata filtering for time-bound data
    - Semantic retrieval caching
    """
    
    def __init__(
        self,
        collection_name: str = "bioask",
        model_name: str = "all-MiniLM-L6-v2",
        host: str = "localhost",
        port: int = 6333,
        shard_number: int = 2,
        enable_quantization: bool = False,
        enable_semantic_cache: bool = True,
        cache_threshold: float = 0.35,
        cache_size_limit: int = 1000
    ):
        self.collection_name = collection_name
        self.model_name = model_name
        self.host = host
        self.port = port
        self.shard_number = shard_number
        self.enable_quantization = enable_quantization
        self._dimension: Optional[int] = None  # Lazy loading
        self._embedding_model: Optional[SentenceTransformer] = None  # Lazy loading
        
        # Initialize Qdrant client with compatibility check disabled
        self.client = QdrantClient(host=host, port=port, check_compatibility=False)
        
        # Semantic cache setup
        if enable_semantic_cache:
            self._setup_semantic_cache(cache_threshold, cache_size_limit)
        else:
            self.semantic_cache = None
    
    def _setup_semantic_cache(self, threshold: float, size_limit: int):
        """Setup in-memory semantic cache for query results."""
        try:
            # Create in-memory cache collection
            self.cache_client = QdrantClient(":memory:")
            self.cache_collection_name = "semantic_cache"
            
            self.cache_client.create_collection(
                collection_name=self.cache_collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # Dimension for all-MiniLM-L6-v2
                    distance=models.Distance.COSINE
                )
            )
            
            self.cache_threshold = threshold
            self.cache_size_limit = size_limit
            self.cache_stats = {"hits": 0, "misses": 0}
            print(f"Semantic cache initialized with threshold {threshold}")
            
        except Exception as e:
            print(f"Failed to initialize semantic cache: {e}")
            self.semantic_cache = None
    
    @property 
    def dimension(self) -> int:
        """Lazy loading of embedding dimension - only calculated when needed."""
        if self._dimension is None:
            print("Loading SentenceTransformer model to get dimension...")
            temp_model = SentenceTransformer(self.model_name)
            self._dimension = temp_model.get_sentence_embedding_dimension()
            del temp_model  # Clean up
            print(f"Embedding dimension: {self._dimension}")
        return self._dimension
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy loading of embedding model - only loaded when needed for retrieval."""
        if self._embedding_model is None:
            print(f"Loading SentenceTransformer model: {self.model_name}")
            self._embedding_model = SentenceTransformer(self.model_name)
        return self._embedding_model
        
    def create_optimized_collection(self):
        """Create collection with optimized HNSW parameters and optional quantization."""
        config = {
            "collection_name": self.collection_name,
            "vectors_config": models.VectorParams(
                size=self.dimension,
                distance=models.Distance.COSINE
            ),
            "optimizers_config": models.OptimizersConfigDiff(
                indexing_threshold=0  # Disable indexing during upload
            ),
            "hnsw_config": models.HnswConfigDiff(
                m=64,  # Increased from default 16 for better precision
                ef_construct=512,  # Increased from default 100 for better recall
                on_disk=True  # Enable disk storage for large datasets
            ),
            "shard_number": self.shard_number
        }
        
        # Add quantization if enabled
        if self.enable_quantization:
            config["quantization_config"] = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True,
                ),
            )
            print("Quantization enabled: 4x memory reduction with minimal precision loss")
        
        self.client.recreate_collection(**config)
        print(f"Created optimized collection '{self.collection_name}' with {self.dimension} dimensions")
        print(f"HNSW config: m=64, ef_construct=512, on_disk=True")
    
    def create_collection(self):
        """Alias for create_optimized_collection for backward compatibility."""
        self.create_optimized_collection()
    
    def setup_metadata_indexes(self):
        """Create indexes for metadata filtering capabilities."""
        try:
            # Index for timestamp-based filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="timestamp",
                field_schema="datetime"
            )
            
            # Index for category-based filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="category",
                field_schema="keyword"
            )
            
            # Index for passage_id for exact lookups
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="passage_id",
                field_schema="keyword"
            )
            
            print("Metadata indexes created for optimized filtering")
            
        except Exception as e:
            print(f"Warning: Could not create metadata indexes: {e}")
    
    def _search_semantic_cache(self, query_embedding: List[float]) -> Optional[List[Dict]]:
        """Search semantic cache for similar queries."""
        if not hasattr(self, 'cache_client'):
            return None
            
        try:
            search_results = self.cache_client.search(
                collection_name=self.cache_collection_name,
                query_vector=query_embedding,
                limit=1,
                with_payload=True
            )
            
            if search_results and search_results[0].score >= (1 - self.cache_threshold):
                self.cache_stats["hits"] += 1
                return search_results[0].payload["results"]
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            print(f"Cache search error: {e}")
            return None
    
    def _add_to_semantic_cache(self, query_embedding: List[float], results: List[Dict]):
        """Add query results to semantic cache."""
        if not hasattr(self, 'cache_client'):
            return
            
        try:
            # Check cache size limit
            collection_info = self.cache_client.get_collection(self.cache_collection_name)
            
            if collection_info.points_count >= self.cache_size_limit:
                # Simple eviction: remove oldest entries
                scroll_result = self.cache_client.scroll(
                    collection_name=self.cache_collection_name,
                    limit=100,
                    with_payload=True
                )
                
                if scroll_result[0]:
                    ids_to_delete = [point.id for point in scroll_result[0]]
                    self.cache_client.delete(
                        collection_name=self.cache_collection_name,
                        points_selector=models.PointIdsList(points=ids_to_delete)
                    )
            
            # Add new cache entry
            point_id = str(uuid.uuid4())
            point = models.PointStruct(
                id=point_id,
                vector=query_embedding,
                payload={
                    "results": results,
                    "timestamp": time.time()
                }
            )
            
            self.cache_client.upsert(
                collection_name=self.cache_collection_name,
                points=[point]
            )
            
        except Exception as e:
            print(f"Cache add error: {e}")
    
    def retrieve_passages(self, query: str, k: int = 5, 
                         use_cache: bool = True,
                         disable_rescoring: bool = False,
                         category_filter: str = None,
                         hours_back: int = None) -> List[Dict]:
        """
        Advanced retrieval with multiple optimization techniques.
        
        Args:
            query: Search query text
            k: Number of passages to retrieve
            use_cache: Whether to use semantic caching
            disable_rescoring: Disable rescoring for faster search (quantization only)
            category_filter: Filter by category metadata
            hours_back: Filter by time (hours from now)
        """
        start_time = time.time()
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Check semantic cache first
        if use_cache and hasattr(self, 'cache_client'):
            cached_results = self._search_semantic_cache(query_embedding)
            if cached_results:
                print(f"Cache hit! Retrieved in {time.time() - start_time:.3f}s")
                return cached_results
        
        # Build filters
        filter_conditions = []
        
        if category_filter:
            filter_conditions.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category_filter)
                )
            )
        
        if hours_back:
            time_threshold = datetime.now() - timedelta(hours=hours_back)
            filter_conditions.append(
                models.FieldCondition(
                    key="timestamp",
                    range=models.DatetimeRange(gte=time_threshold.isoformat())
                )
            )
        
        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None
        
        # Configure search parameters
        search_params = None
        if self.enable_quantization and disable_rescoring:
            search_params = models.SearchParams(
                quantization=models.QuantizationSearchParams(rescore=False)
            )
        
        # Perform search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=k,
            search_params=search_params,
            with_payload=True,
            with_vectors=False
        )
        
        # Format results
        retrieved_passages = []
        for result in search_results:
            retrieved_passages.append({
                'text': result.payload.get('text', ''),
                'id': result.payload.get('passage_id', 'unknown'),
                'score': result.score,
                'metadata': {
                    'qdrant_id': result.id,
                    'collection_name': self.collection_name
                }
            })
        
        # Add to cache
        if use_cache and hasattr(self, 'cache_client'):
            self._add_to_semantic_cache(query_embedding, retrieved_passages)
        
        search_time = time.time() - start_time
        print(f"Retrieved {len(retrieved_passages)} passages in {search_time:.3f}s")
        
        return retrieved_passages
    
    def fast_retrieve_passages(self, query: str, k: int = 5) -> List[Dict]:
        """Fast retrieval with disabled rescoring (quantization must be enabled)."""
        return self.retrieve_passages(
            query=query, 
            k=k, 
            use_cache=True, 
            disable_rescoring=True
        )
    
    def time_aware_retrieve(self, query: str, hours_back: int = 24, 
                           category: str = None, k: int = 5) -> List[Dict]:
        """Time-aware retrieval for temporal relevance."""
        return self.retrieve_passages(
            query=query,
            k=k,
            category_filter=category,
            hours_back=hours_back
        )
    
    def get_cache_statistics(self) -> Dict:
        """Get semantic cache performance statistics."""
        if not hasattr(self, 'cache_stats'):
            return {"cache_enabled": False}
            
        total_queries = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_ratio = self.cache_stats["hits"] / total_queries if total_queries > 0 else 0
        
        cache_size = 0
        if hasattr(self, 'cache_client'):
            try:
                collection_info = self.cache_client.get_collection(self.cache_collection_name)
                cache_size = collection_info.points_count
            except:
                cache_size = 0
        
        return {
            "cache_enabled": True,
            "hit_ratio": hit_ratio,
            "total_hits": self.cache_stats["hits"],
            "total_misses": self.cache_stats["misses"],
            "cache_entries": cache_size,
            "cache_size_limit": self.cache_size_limit,
            "similarity_threshold": self.cache_threshold
        }
    
    def get_collection_info(self) -> Dict:
        """Get information about the current collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
        except Exception as e:
            return {"error": str(e)}
    
    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except:
            return False
    