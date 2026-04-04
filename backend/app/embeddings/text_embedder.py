from sentence_transformers import SentenceTransformer
import numpy as np

from app.embeddings.base_embedder import BaseEmbedder

class TextEmbedder(BaseEmbedder):
    """
    Embedder for text using SentenceTransformer.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the TextEmbedder.
        """
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of texts to encode.

        Returns:
            Numpy array of embeddings.
        """
        return self.model.encode(texts, convert_to_numpy=True)

    