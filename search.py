import os
import json
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Step 1: load our saved embeddings
with open("embeddings.json", "r", encoding="utf-8") as f:
    embedded_data = json.load(f)

# Step 2: embed the user's question the same way we embedded chunks
def embed_query(query):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
    return result.embeddings[0].values

# Step 3: cosine similarity — measures how close two embeddings are
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Step 4: search — find the top matching chunks for a question
def search(query, top_k=3):
    query_embedding = embed_query(query)
    scored_chunks = []
    for item in embedded_data:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored_chunks.append((score, item["text"]))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return scored_chunks[:top_k]

# Test it
question = "How do I undo changes in git?"
results = search(question)

for score, text in results:
    print(f"Score: {score:.4f}")
    print(text[:300])
    print("---")