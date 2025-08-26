import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# --- ベクトルストアの読み込み ---
@st.cache_resource
def load_vectorstore(api_key: str = None, FAISS_PATH: str = None):
    if not api_key:
        st.warning("OPENAI_API_KEY が未設定のためベクトルストアをロードできません。")
        return None
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.load_local(
        FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True
    )
