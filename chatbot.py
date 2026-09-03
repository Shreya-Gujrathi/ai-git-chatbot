import os
import json
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Load our saved knowledge base
with open("embeddings.json", "r", encoding="utf-8") as f:
    embedded_data = json.load(f)

def embed_query(query):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
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
    # Step 1: Retrieve relevant chunks
    top_chunks = search(question)
    context = "\n\n".join([text for score, text in top_chunks])

    # Step 2: Build a prompt combining context + question
    prompt = f"""You are a helpful assistant that answers questions about Git, using ONLY the context provided below. If the answer isn't in the context, say you don't know based on the available information.

Context:
{context}

Question: {question}

Answer:"""

    # Step 3: Generate the answer
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text

# Test it
print("=== AI Chatbot (type 'quit' to exit) ===")
while True:
    user_question = input("\nYour question: ")
    if user_question.lower() == "quit":
        break
    answer = ask_chatbot(user_question)
    print(f"\nBot: {answer}")