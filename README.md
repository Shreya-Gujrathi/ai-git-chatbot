# 📘 Git Docs Chatbot — RAG-based AI Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Git using the official open-source **Pro Git** book as its knowledge base. Built from scratch, deployed live, with an automated CI pipeline.

🔗 **Live app:** https://ai-git-chatbot-mflaqhhiugwzbgur5r5y7u.streamlit.app
📂 **Repo:** https://github.com/Shreya-Gujrathi/ai-git-chatbot

---

## What it does

Ask any question about Git, and the chatbot retrieves the most relevant passages from the *Pro Git* book and generates a grounded answer — instead of relying purely on the AI model's general training. If the answer isn't in the source material, it says so rather than guessing.

## How it works (RAG pipeline)

1. **Extraction** — Text is extracted from the Pro Git PDF using `pypdf`
2. **Chunking** — Extracted text is split into overlapping ~1000-character chunks to preserve context
3. **Embedding** — Each chunk is converted into a 3072-dimension vector using Google's `gemini-embedding-001` model, capturing semantic meaning
4. **Retrieval** — When a question is asked, it's embedded the same way, then compared against all chunk embeddings using **cosine similarity** to find the most relevant matches
5. **Generation** — The top matching chunks are passed to Gemini (`gemini-3.6-flash`) as context, along with a prompt instructing the model to answer *only* from that context
6. **Interface** — Built with **Streamlit** for a clean, chat-style UI

## Tech stack

- **Language:** Python
- **AI Model:** Google Gemini (`gemini-3.6-flash` for generation, `gemini-embedding-001` for embeddings)
- **PDF Processing:** pypdf
- **Similarity Search:** NumPy (cosine similarity)
- **UI:** Streamlit
- **CI/CD:** GitHub Actions (automated dependency install + syntax checks on every push) + Streamlit Community Cloud (automatic redeploy on push to `main`)
- **Version Control:** Git & GitHub

## Project structure

```
├── app.py              # Main application (RAG pipeline + Streamlit UI)
├── embed_all.py         # One-time script to build the knowledge base (extract → chunk → embed)
├── embeddings.json      # Saved knowledge base (chunks + embeddings)
├── requirements.txt      # Python dependencies
├── .github/workflows/ci.yml   # CI pipeline configuration
└── .gitignore
```

## Running it locally

```bash
git clone https://github.com/Shreya-Gujrathi/ai-git-chatbot.git
cd ai-git-chatbot
pip install -r requirements.txt
```

Create a `.env` file with your own Gemini API key:
```
GEMINI_API_KEY=your_key_here
```

Then run:
```bash
streamlit run app.py
```

## Notes

- The knowledge base currently covers the early chapters of *Pro Git* — this can be expanded to the full book by adjusting the page range in `embed_all.py`.
- The Pro Git book is freely available under a Creative Commons license (Scott Chacon & Ben Straub).
