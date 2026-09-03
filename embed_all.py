import os
import json
import time
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Step 1: extract + chunk (same as before)
reader = PdfReader("progit.pdf")
text = ""
for page_num in range(15, 65):
    text += reader.pages[page_num].extract_text()

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = chunk_text(text)
print(f"Total chunks to embed: {len(chunks)}")

# Step 2: embed every chunk, one at a time
embedded_data = []
for i, chunk in enumerate(chunks):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    )
    embedding = result.embeddings[0].values
    embedded_data.append({
        "id": i,
        "text": chunk,
        "embedding": embedding
    })
    print(f"Embedded chunk {i+1}/{len(chunks)}")
    time.sleep(1)  # small pause to avoid hitting rate limits

# Step 3: save everything to a file so we don't have to re-embed every time
with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embedded_data, f)

print("All chunks embedded and saved to embeddings.json")