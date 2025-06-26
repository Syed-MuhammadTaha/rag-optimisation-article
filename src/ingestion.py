from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import time
import hashlib
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class CachedEmbeddingModel:
    """Embedding model with caching support for improved ingestion performance."""
    
    def __init__(self, model_name: str, cache_file: str = "embedding_cache.pkl"):
        self.model = SentenceTransformer(model_name)
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _load_cache(self) -> Dict[str, List[float]]:
        """Load existing embedding cache from disk."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                print(f"Loaded embedding cache with {len(cache)} entries")
                return cache
        except Exception as e:
            print(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"Saved embedding cache with {len(self.cache)} entries")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts with caching support - directly improves ingestion speed."""
        if isinstance(texts, str):
            texts = [texts]
        
        start_time = time.time()
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text (very fast lookup)
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            if text_hash in self.cache:
                embeddings.append(self.cache[text_hash])
                self.cache_hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                embeddings.append(None)  # Placeholder
                self.cache_misses += 1
        
        # Only compute embeddings for uncached texts (saves computation time)
        if uncached_texts:
            print(f"Computing embeddings for {len(uncached_texts)} new texts...")
            new_embeddings = self.model.encode(uncached_texts).tolist()
            
            # Update cache and results
            for idx, embedding in zip(uncached_indices, new_embeddings):
                text_hash = self._get_text_hash(texts[idx])
                self.cache[text_hash] = embedding
                embeddings[idx] = embedding
            
            # Save cache periodically (every 100 new embeddings)
            if len(uncached_texts) > 0:
                self._save_cache()
        
        total_time = time.time() - start_time
        cache_ratio = self.cache_hits / (self.cache_hits + self.cache_misses) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0
        
        print(f"Embedding cache performance: {self.cache_hits} hits, {self.cache_misses} misses ({cache_ratio:.1f}% hit rate)")
        print(f"Embedding computation time: {total_time:.2f}s")
        
        return embeddings
    
    def get_sentence_embedding_dimension(self):
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


class DocumentIngestion:
    """Handles document ingestion into Qdrant vector store with optimization."""
    
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 100,
        max_parallel_batches: int = 4,
        cache_file: str = "embedding_cache.pkl"
    ):
        self.client = client
        self.collection_name = collection_name
        self.model = CachedEmbeddingModel(model_name, cache_file)
        self.batch_size = batch_size
        self.max_parallel_batches = max_parallel_batches
    
    def _process_batch(self, batch_data: Tuple[List[Dict], int]) -> List[models.PointStruct]:
        """Process a single batch of passages with global ID tracking and caching."""
        batch, start_idx = batch_data
        
        # Encode all texts in batch at once with caching support
        texts = [p["text"] for p in batch]
        embeddings = self.model.encode(texts)  # This now uses caching
        
        points = []
        for i, (passage, embedding) in enumerate(zip(batch, embeddings)):
            global_id = start_idx + i  # Use global ID across all batches
            point = models.PointStruct(
                id=global_id,
                vector=embedding,
                payload={
                    "text": passage["text"],
                    "id": passage["id"]
                }
            )
            points.append(point)
        return points

    def _insert_batch(self, points: List[models.PointStruct]) -> None:
        """Insert a batch of points into Qdrant."""
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=False  # Async insertion for better performance
        )

    def insert_passages(self, passages: List[Dict], limit: int = 2500) -> Tuple[float, List[int]]:
        """
        Insert passages into vector store using parallel batch processing with embedding cache.
        
        Args:
            passages: List of passage dictionaries
            limit: Maximum number of passages to process
            
        Returns:
            tuple: (total_time, list of processed IDs)
        """
        start_time = time.time()
        total_passages = min(len(passages), limit)
        processed_ids = []
        
        # Split passages into batches with global ID tracking
        batch_data = []
        for i in range(0, total_passages, self.batch_size):
            batch = passages[i:i + self.batch_size]
            start_idx = i + 1  # Global starting ID for this batch
            batch_data.append((batch, start_idx))
        
        print(f"Processing {total_passages} passages in {len(batch_data)} batches")
        print(f"Batch size: {self.batch_size}, Expected total points: {total_passages}")
        print(f"Using embedding cache: {self.model.cache_file}")
        
        with ThreadPoolExecutor(max_workers=self.max_parallel_batches) as executor:
            # Process batches in parallel
            batch_points = list(tqdm(
                executor.map(self._process_batch, batch_data),
                total=len(batch_data),
                desc="Processing batches"
            ))
            
            # Insert processed batches
            for points in tqdm(batch_points, desc="Inserting batches"):
                self._insert_batch(points)
                processed_ids.extend([p.id for p in points])
        
        total_time = time.time() - start_time
        print(f"Successfully inserted {len(processed_ids)} points")
        
        # Print final cache statistics
        total_requests = self.model.cache_hits + self.model.cache_misses
        if total_requests > 0:
            cache_efficiency = (self.model.cache_hits / total_requests) * 100
            print(f"Final cache efficiency: {cache_efficiency:.1f}% ({self.model.cache_hits}/{total_requests})")
        
        # Re-enable indexing after upload
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000  # Default value
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16  # Standard M value
            )
        )
        
        return total_time, processed_ids 