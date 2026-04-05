class VectorStore:
    """
    Vector store for storing and retrieving embeddings.
    """
    def __init__(self, dim: int) -> None:
        """
        Initialize the VectorStore.
        """
        self.index = faiss.IndexFlatL2(dim)
        self.metadata: list[dict[str, Any]] = []
    
    def add(self, embeddings: np.ndarray, data: list[dict[str, Any]]) -> None:
        """
        Add texts to the vector store.
        """
        self.index.add(embeddings)
        self.metadata.extend(data)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        distances, indices = self.index.search(query_embedding, k)
        results = []

        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                "metadata": self.metadata[idx],
                "distance": dist
            })

        return results