from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class VectorStore:
    def __init__(
        self,
        collection_name: str = "bioasq",
        model_name: str = "all-MiniLM-L6-v2",
        host: str = "localhost",
        port: int = 6333,
        batch_size: int = 100,  # Standard batch size
        max_parallel_batches: int = 4,  # Number of parallel batches
        shard_number: int = 2  # Number of shards for parallel processing
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.max_parallel_batches = max_parallel_batches
        self.shard_number = shard_number
        
    def create_collection(self):
        """Create a collection with optimized settings for bulk upload."""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dimension,
                distance=models.Distance.COSINE
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0  # Disable indexing during upload
            ),
            hnsw_config=models.HnswConfigDiff(
                m=0,  # Defer HNSW graph construction
                on_disk=True
            ),
            shard_number=self.shard_number  # Enable parallel processing with shards
        )
    
    def _process_batch(self, batch: List[Dict]) -> List[models.PointStruct]:
        """Process a single batch of passages."""
        # Encode all texts in batch at once for better performance
        texts = [p["text"] for p in batch]
        embeddings = self.model.encode(texts)
        
        points = []
        for idx, (passage, embedding) in enumerate(zip(batch, embeddings), 1):
            point = models.PointStruct(
                id=idx,
                vector=embedding.tolist(),
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
        Insert passages into vector store using parallel batch processing.
        
        Args:
            passages: List of passage dictionaries
            limit: Maximum number of passages to process
            
        Returns:
            tuple: (total_time, list of processed IDs)
        """
        start_time = time.time()
        total_passages = min(len(passages), limit)
        processed_ids = []
        
        # Split passages into batches
        batches = [
            passages[i:i + self.batch_size] 
            for i in range(0, total_passages, self.batch_size)
        ]
        
        with ThreadPoolExecutor(max_workers=self.max_parallel_batches) as executor:
            # Process batches in parallel
            batch_points = list(tqdm(
                executor.map(self._process_batch, batches),
                total=len(batches),
                desc="Processing batches"
            ))
            
            # Insert processed batches
            for points in tqdm(batch_points, desc="Inserting batches"):
                self._insert_batch(points)
                processed_ids.extend([p.id for p in points])
        
        total_time = time.time() - start_time
        
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
    