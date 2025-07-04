import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import openai
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter

# --- CONFIG ---
TEXT_COLUMN = "Addiction_Reason"  # <-- ✅ Choose a meaningful text column
DATA_PATH = "Students Social Media Addiction.csv"  # <-- ✅ Just the filename
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# --- API KEY ---
openai.api_key = os.getenv("5d09bf40b212e01aa6f740e3e586a957932303117b43caf80bac2db0372d3989")  # Hugging Face Secret
openai.api_base = "https://api.together.xyz/v1"

if not openai.api_key:
    st.error("❌ TOGETHER_API_KEY not found. Please set it in Hugging Face Space secrets.")
    st.stop()

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if TEXT_COLUMN not in df.columns:
        st.error(f"Column '{TEXT_COLUMN}' not found in dataset.")
        st.stop()
    text_data = "\n".join(df[TEXT_COLUMN].dropna().astype(str).tolist())
    return text_data

# --- Embed + Index ---
@st.cache_resource
def setup_index(text_data):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(text_data)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return docs, index, model

# --- Retrieve Docs ---
def retrieve_docs(query, docs, index, model, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [docs[i] for i in indices[0]]

# --- LLM Query ---
def call_llm(context, query):
    prompt = f"""Answer the user's question using only the context below:

Context:
{context}

Question:
{query}

Answer:"""
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

# --- STREAMLIT UI ---
st.set_page_config(page_title="Student Addiction Chatbot", layout="centered")
st.title("📚 Student Social Media Addiction Chatbot")
st.markdown("Ask questions about **social media addiction in students**. This chatbot uses Retrieval-Augmented Generation (RAG).")

query = st.text_input("💬 Ask your question")

if query:
    with st.spinner("Searching and thinking..."):
        text_data = load_data()
        docs, index, model = setup_index(text_data)
        retrieved = retrieve_docs(query, docs, index, model)
        context = "\n".join(retrieved)
        response = call_llm(context, query)
        st.success(response)
