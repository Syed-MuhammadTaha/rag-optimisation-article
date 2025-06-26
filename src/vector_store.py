from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# LangChain imports for retrieval
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document


class VectorStore:
    """
    Qdrant vector store focused on configuration, setup, and retrieval using LangChain.
    """
    
    def __init__(
        self,
        collection_name: str = "bioasq",
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
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=host, port=port)
        
        # Get embedding dimension
        temp_model = SentenceTransformer(model_name)
        self.dimension = temp_model.get_sentence_embedding_dimension()
        del temp_model  # Clean up
        
        # Initialize LangChain embeddings
        self.embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        
    def create_collection(self):
        """Create a Qdrant collection with optimized settings."""
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
        print(f"Created collection '{self.collection_name}' with {self.dimension} dimensions")
    
    def setup_langchain_vectorstore(self) -> QdrantVectorStore:
        """
        Setup LangChain Qdrant vector store for retrieval.
        
        Returns:
            QdrantVectorStore: LangChain vector store instance
        """
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
        )
        return vector_store
    
    def retrieve_passages(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most similar passages for a given query using LangChain.
        
        Args:
            query: Search query text
            k: Number of passages to retrieve
            
        Returns:
            List of dictionaries containing passage information and scores
        """
        vector_store = self.setup_langchain_vectorstore()
        
        # Perform similarity search with scores
        results = vector_store.similarity_search_with_score(query, k=k)
        
        # Format results
        retrieved_passages = []
        for doc, score in results:
            retrieved_passages.append({
                'text': doc.page_content,
                'id': doc.metadata.get('id', 'unknown'),
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
    