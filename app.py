import os
import json
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

with open("embeddings.json", "r", encoding="utf-8") as f:
    embedded_data = json.load(f)

def embed_query(query):
    result = client.models.embed_content(model="gemini-embedding-001", contents=query)
    return result.embeddings[0].values

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query, top_k=3):
    query_embedding = embed_query(query)
    scored_chunks = []
    for item in embedded_data:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored_chunks.append((score, item["text"]))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return scored_chunks[:top_k]

def ask_chatbot(question):
    top_chunks = search(question)
    context = "\n\n".join([text for score, text in top_chunks])
    prompt = f"""You are a helpful assistant that answers questions about Git, using ONLY the context provided below. If the answer isn't in the context, say you don't know based on the available information.

Context:
{context}

Question: {question}

Answer:"""
    response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
    return response.text

# ---- UI starts here ----
st.title("📘 Git Docs Chatbot")
st.write("Ask me anything about Git — I answer using the Pro Git book.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask a Git question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_chatbot(user_input)
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})