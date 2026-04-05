from typing import Any

from app.embeddings.base_embedder import BaseEmbedder
from app.retriever.vector_store import VectorStore

class SearchService:
    """
    Service for searching the vector store.
    """
    def __init__(self, embedder: BaseEmbedder, vector_store: VectorStore) -> None:
        """
        Initialize the SearchService.
        """
        self.embedder = embedder
        self.vector_store = vector_store
    
    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """
        Search the vector store for the query.
        """
        query_embedding = self.embedder.encode([query])
        results = self.vector_store.search(query_embedding, k)
        return results