from app.embeddings.text_embedder import TextEmbedder
from app.embeddings.base_embedder

def get_embedder(embedder_type: str = "text") -> BaseEmbedder:
    """
    Get the appropriate embedder based on the type.
    """
    if embedder_type == "text":
        return TextEmbedder()
    else:
        raise ValueError(f"Invalid embedder type: {embedder_type}")