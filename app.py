import streamlit as st
import os

from database import Database
from router import Router
from generate import Generate
from retrieval import Retrieval
from config import Args

from streamlit_chat import message

st.set_page_config(page_title="Chatbot UI", page_icon="🤖")
st.title("💬 Kicap Store")

args = Args()
os.environ["OPENAI_API_KEY"] = args.openai_api_key


if "database" not in st.session_state:
    st.session_state.database = Database(
        db_uri=args.uri,
        db_name=args.db_name,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        device=args.device
    )
    st.session_state.documents = st.session_state.database.get_all_documents()

if "retriever" not in st.session_state:
    st.session_state.retriever = Retrieval(
        documents=st.session_state.documents,
        embedding_model=args.embedding_model,
        device=args.device
    )
    st.session_state.retriever.build()

if "router" not in st.session_state:
    st.session_state.router = Router()

if "chatbot" not in st.session_state:
    st.session_state.chatbot = Generate(st.session_state.retriever, st.session_state.router)

def get_response(prompt, chatbot):
    response, _ = chatbot.generate_answer(prompt, args.top_k_document)
    return response

if "messages" not in st.session_state:
    st.session_state.messages = []

for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=(msg["role"] == "user"), key=f"msg_{i}")

prompt = st.chat_input("Nhập tin nhắn...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    message(prompt, is_user=True, key=f"user_{len(st.session_state.messages)}")

    response = get_response(prompt, st.session_state.chatbot)
    st.session_state.messages.append({"role": "assistant", "content": response})
    message(response, is_user=False, key=f"bot_{len(st.session_state.messages)}")

if st.button("Xóa lịch sử chat"):
    st.session_state.messages = []
    st.rerun()