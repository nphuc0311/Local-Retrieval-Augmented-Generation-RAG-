from typing import List, Dict
from sentence_transformers import SentenceTransformer

class Embedding:
  def __init__(self, embedding_model: str, device: str = "cpu"):
    """Initialize the embedding model."""
    self.model = SentenceTransformer(embedding_model, device=device)

  def get_embedding(self, text: str) -> List[float]:
    """
    Generate embeddings for the input text.

    Args:
        text (str): The input text.

    Returns:
        List[float]: A list of embeddings.
    """
    if isinstance(text, str):
        text = [text]
    return self.model.encode(text).tolist()