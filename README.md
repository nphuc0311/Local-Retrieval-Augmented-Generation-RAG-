# 💬 Kicap Store Chatbot (Local RAG with Streamlit & LangChain)

Welcome to **Kicap Store Chatbot**, a Local Retrieval-Augmented Generation (RAG) chatbot designed to assist customers of the Kicap mechanical keyboard store. This chatbot provides product recommendations and answers general inquiries, leveraging a hybrid search engine (FAISS + BM25) and a local LLM interface powered by LangChain.

## 🚀 Features

- ✅ **Hybrid Search (FAISS + BM25):** Combines vector search with semantic search for accurate responses.  
- ✅ **Localized in Vietnamese:** Responds naturally in Vietnamese.  
- ✅ **Contextual Conversations:** Maintains chat history for better interactions.  
- ✅ **Streamlit UI:** Simple and interactive user interface.  
- ✅ **Efficient Caching:** Uses `TTLCache` for faster repeated queries.  
- ✅ **Modular Architecture:** Clear separation of responsibilities for easy maintenance.  

---

## 🏗️ Project Structure

```plaintext
📂 kicap-store-chatbot/
├── app.py                # Streamlit UI for chatbot
├── config.py             # Configuration settings
├── database.py           # MongoDB operations & CSV loading
├── embedding.py          # SentenceTransformer embeddings
├── retrieval.py          # Hybrid search (FAISS + BM25)
├── router.py             # Query classification using LangChain
└── generate.py           # Response generation with LLM
```

---

## 🛠️ Installation

### 📌 Prerequisites
- Python 3.10+
- MongoDB
- FAISS

### 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🛡️ Set Environment Variables
Create a `.env` file and add your credentials:
```env
MONGO_URI="your-mongodb-uri"
OPENAI_API_KEY="your-openai-api-key"
```

---

## 🚀 Usage
### **Run the Chatbot**
```bash
streamlit run app.py
```
---

## 🧩 How It Works

### 1️⃣ **Data Preparation (`database.py`):**
- Loads product descriptions from a CSV file.
- Generates embeddings using `SentenceTransformer`.
- Stores documents in MongoDB.

### 2️⃣ **Query Routing (`router.py`):**
- Classifies questions into:
  - 📌 `rag_query`: Product recommendations or comparisons.  
  - 📌 `general_query`: Store policies or usage guides.  

### 3️⃣ **Hybrid Search (`retrieval.py`):**
- ✅ **Dense Search (FAISS):** Finds similar products using embeddings.  
- ✅ **Sparse Search (BM25):** Matches keywords using `rank_bm25`.  
- ✅ **Hybrid Fusion:** Combines results using a weighted score.  

### 4️⃣ **Response Generation (`generate.py`):**
- Uses **LangChain** to generate responses.  
- Maintains conversation history with `ConversationBufferMemory`.  

### 5️⃣ **User Interface (`app.py`):**
- Built with **Streamlit** and `streamlit_chat` for an intuitive chat experience.  

---

## 📝 Future Improvements
- [ ] Add a **Recommendation System** based on user preferences.  
- [ ] Implement a **Feedback Loop** to improve responses.  
- [ ] Add **Docker Support** for easy deployment.  

---
