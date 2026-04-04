from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedder(ABC):
    """
    Abstract base class for embedders.
    """
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        """
        pass