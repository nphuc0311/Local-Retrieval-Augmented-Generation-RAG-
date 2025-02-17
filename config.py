from dataclasses import dataclass


@dataclass
class Args:
  # Mongo
  uri: str = "MONGO URI"
  db_name: str = "kicap"
  collection_name: str = "keyboard"

  # # Langchain API keys
  openai_api_key: str = "OPENAI API KEY"

  # Embedding model
  embedding_model: str = "keepitreal/vietnamese-sbert"

  # Memory settings
  max_history: int = 5
  memory_type: str = "buffer_window"  # buffer | buffer_window | summary

  # Hybrid search
  top_k_document = 5

  # Device
  device: str = "cpu"