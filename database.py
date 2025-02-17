import re
import pandas as pd

from typing import List, Dict
from pymongo import MongoClient

from embedding import Embedding

class Database(Embedding):
    def __init__(self,
                 db_uri: str,
                 db_name: str,
                 collection_name: str,
                 embedding_model: str,
                 device: str = "cpu"):
      """
      Initialize the database connection and embedding model.

      Args:
          db_uri (str): MongoDB connection URI.
          db_name (str): Database name.
          collection_name (str): Collection name.
          embedding_model (str): Embedding model to use.
          device (str): Device to use for embeddings (default: "cpu").
      """
      super().__init__(embedding_model, device)
      self.client = MongoClient(db_uri)
      self.db = self.client[db_name]
      self.collection = self.db[collection_name]
      self.data = []

    @staticmethod
    def preprocess_text(text: str) -> str:
      """
      Preprocess the input text by converting to lowercase,
      removing special characters, and extra spaces.

      Args:
          text (str): The input text.

      Returns:
          str: Cleaned and preprocessed text.
      """
      text = text.lower()  # Convert to lowercase
      text = re.sub(r'[^\w\s,.]', '', text)  # Remove special characters
      text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
      return text

    def load_csv(self, path: str) -> None:
      """
      Load a CSV file, preprocess its content, and store it as a list of dictionaries.

      Args:
          path (str): Path to the CSV file.
      """
      print(f"Loading CSV file at {path}")
      df = pd.read_csv(path, encoding="utf-8")
      # Drop unnecessary columns
      df.drop(columns=["link-products", "web-scraper-order", "web-scraper-start-url"], inplace=True)
      df.rename(columns={'link-products-href': 'url product'}, inplace=True)

      # Preprocess relevant columns
      columns_to_apply = df.drop(columns=["url product"]).columns
      for col in columns_to_apply:
          df[col] = df[col].map(self.preprocess_text)

      self.data = df.to_dict('records')
      print("Processing CSV file done")

    def insert_document(self) -> None:
      """
      Insert preprocessed data with embeddings into MongoDB.
      """
      print("Inserting documents into MongoDB.")
      for i in range(len(self.data)):
        text = f'{self.data[i]["title"]} {self.data[i]["description"]}'
        self.data[i]["embedding"] = self.get_embedding(text)

      self.collection.insert_many(self.data)
      print("Documents inserted into MongoDB successfully.")

    def get_all_documents(self) -> List[Dict]:
      """
      Retrieve all documents from the MongoDB collection.

      Returns:
          List[Dict]: A list of documents with metadata and embeddings.
      """
      return list(self.collection.find({}, {"_id": 0}))
