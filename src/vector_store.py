from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class VectorStore:
    def __init__(
        self,
        collection_name: str = "bioasq",
        model_name: str = "all-MiniLM-L6-v2",
        host: str = "localhost",
        port: int = 6333
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def create_collection(self):
        """Create a new collection in Qdrant."""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dimension,
                distance=models.Distance.COSINE
            )
        )
    
    def insert_passages(self, passages: List[Dict]):
        """Insert passages into the vector store."""
        texts = [p["text"] for p in passages]
        embeddings = self.model.encode(texts)
        
        points = [
            models.PointStruct(
                id=p["id"],
                vector=embedding.tolist(),
                payload={
                    "text": p["text"],
                    "id": p["id"]  # Adding ID to metadata
                }
            )
            for p, embedding in zip(passages, embeddings)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar passages."""
        query_vector = self.model.encode(query).tolist()
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        return [
            {
                "id": str(r.id),
                "text": r.payload["text"],
                "score": r.score
            }
            for r in results
        ] 