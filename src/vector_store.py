from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

# LangChain imports for retrieval
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


class VectorStore:
    """
    Qdrant vector store focused on configuration, setup, and retrieval using LangChain.
    """
    
    def __init__(
        self,
        collection_name: str = "bioask",
        model_name: str = "all-MiniLM-L6-v2",
        host: str = "localhost",
        port: int = 6333,
        shard_number: int = 2
    ):
        self.collection_name = collection_name
        self.model_name = model_name
        self.host = host
        self.port = port
        self.shard_number = shard_number
        self._dimension: Optional[int] = None  # Lazy loading
        
        # Initialize Qdrant client with compatibility check disabled
        self.client = QdrantClient(host=host, port=port, check_compatibility=False)
        
        # Initialize LangChain embeddings using HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Initialize LangChain vector store directly in constructor
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name=self.collection_name,
            url=f"http://{host}:{port}",
            content_payload_key="text"  # Map payload 'text' field to page_content
        )
    
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
        
    def create_collection(self):
        """Create a Qdrant collection with optimized settings."""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dimension,  # This will trigger lazy loading if needed
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
        print(f"Created collection '{self.collection_name}' with {self.dimension} dimensions")
    
    def retrieve_passages(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most similar passages for a given query using LangChain.
        
        Args:
            query: Search query text
            k: Number of passages to retrieve
            
        Returns:
            List of dictionaries containing passage information and scores
        """
        # Perform similarity search with scores using the pre-configured vector store
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Format results
        retrieved_passages = []
        for doc, score in results:
            # Get the original ID from the vector store payload
            original_id = None
            if hasattr(doc, 'metadata') and doc.metadata.get('_id'):
                # Try to get the original ID from Qdrant by fetching the point
                try:
                    point = self.client.retrieve(
                        collection_name=self.collection_name,
                        ids=[doc.metadata['_id']],
                        with_payload=True
                    )[0]
                    original_id = point.payload.get('id', 'unknown')
                except:
                    original_id = 'unknown'
            
            retrieved_passages.append({
                'text': doc.page_content,
                'id': original_id,
                'score': score,
                'metadata': doc.metadata
            })
        
        return retrieved_passages
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary with collection information
        """
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
    