import numpy as np
import faiss

from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache

from embedding import Embedding


class Retrieval(Embedding):
  def __init__(self,
               documents: List[Dict],
               embedding_model: str,
               device: str = "cpu"):

    super().__init__(embedding_model, device)

    if not documents:
      raise ValueError("No documents provided.")

    self.documents = documents
    self.index = None
    self.cache = TTLCache(maxsize=100, ttl=3600)


  def build(self) -> None:
    """Build a FAISS index."""

    embeddings = np.array([doc["embedding"] for doc in self.documents]).astype("float32").squeeze()
    dim = embeddings.shape[1]
    normallized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    self.index = faiss.IndexFlatIP(dim)
    self.index.add(normallized_embeddings)

    print("FAISS index built and loaded from MongoDB.")


  def vector_search(self, query, top_k=5) -> List[Tuple]:
    similarities, indices = self.index.search(query, k=top_k)
    return [(idx, similarities[0][i]) for i, idx in enumerate(indices[0])]


  def text_search(self, query, top_k=5):
    corpus = [doc["title"] + " " + doc["description"] for doc in self.documents]
    corpus = [ViTokenizer.tokenize(text).split() for text in corpus]
    bm25 = BM25Okapi(corpus)

    query = ViTokenizer.tokenize(query).split()
    scores = bm25.get_scores(query)
    min_score = min(scores)
    max_score = max(scores)
    scores_scaled = [(score - min_score) / (max_score - min_score) if max_score > min_score else 0 for score in scores]
    scores = [(idx, score) for idx, score in enumerate(scores_scaled)]
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_k]



  def hybrid_search(self, query, top_k=5, alpha=0.7):
    """Perform hybrid search by combining dense and sparse results."""
    # Check cache
    if query in self.cache:
      return self.cache[query]

    # Embed query
    query_embedding = self.get_embedding(query)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    # Parallelize dense and sparse retrieval
    with ThreadPoolExecutor() as executor:
      vector_future = executor.submit(self.vector_search, query_embedding, top_k)
      text_future = executor.submit(self.text_search, query, top_k)

      vector_results = vector_future.result()
      text_results = text_future.result()

    # Combine and normalize scores
    combined_results = {}
    for idx, score in vector_results:
      combined_results[idx] = combined_results.get(idx, 0) + score * alpha
    for idx, score in text_results:
      combined_results[idx] = combined_results.get(idx, 0) + score * (1 - alpha)

    # Sort combined results by score
    ranked_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
    ranked_results = [self.documents[idx] for idx, scores in ranked_results]

    # Cache the result
    ranked_results = [{k: v for k, v in d.items() if k != "embedding"} for d in ranked_results]

    self.cache[query] = ranked_results[:top_k]
    return ranked_results[:top_k]

  def search(self, query, top_k=5):
    """Perform hybrid search by default."""
    return self.hybrid_search(query, top_k)

