# Campus RAG Assistant

A Retrieval-Augmented Generation (RAG) application that lets students upload their course notes (PDFs) and ask natural-language questions about them, with answers grounded in the uploaded material and cited by source page.

## Overview

Instead of manually skimming through dozens of pages of notes before an exam, a student uploads their PDFs and asks direct questions. The system retrieves the most relevant chunks of text using vector search, then generates an answer using an LLM — with citations back to the exact source document and page.

## Features

- PDF upload and automatic text chunking
- Semantic search over notes using vector embeddings
- Grounded, cited question-answering (no hallucinated answers outside the uploaded content)
- Simple chat-style web interface
- Fully containerized with Docker

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers (`BAAI/bge-small-en-v1.5`) |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Frontend | Streamlit |
| Containerization | Docker |

## Architecture
PDF Upload → Text Extraction (PyMuPDF) → Chunking → Embedding → ChromaDB
↓
User Question → Embedding → Vector Search ← ─────────────────────────┘
↓
Top-k relevant chunks
↓
LLM Prompt (context + question) → Groq API → Cited Answer


## Setup

### Prerequisites
- Python 3.12
- Docker Desktop
- A Groq API key ([console.groq.com](https://console.groq.com))

### Local setup (without Docker)
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:
GROQ_API_KEY=your_key_here


Run the backend:
```bash
uvicorn app.main:app --reload
```

Run the UI (in a separate terminal):
```bash
streamlit run app/ui.py
```

### Running with Docker
```bash
docker build -t campus-rag-assistant .
docker run -p 7860:7860 --env-file .env campus-rag-assistant
```

Then run the Streamlit UI separately, pointing to the container:
```bash
streamlit run app/ui.py
```
(Make sure `app/ui.py`'s `API` variable matches the container's port, `http://127.0.0.1:7860`.)

## Usage

1. Open the Streamlit UI in your browser.
2. Upload a PDF of your notes using the sidebar.
3. Wait for it to be processed and added to the knowledge base.
4. Type a question in the main chat box.
5. The answer appears along with the source document and page number it was drawn from.

### API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/upload` | Upload and ingest a PDF |
| POST | `/ask` | Ask a question, returns answer + sources |

Interactive API docs available at `/docs` when the backend is running.

## Deployment Status

The application is fully functional and tested end-to-end in Docker, locally. Cloud deployment was attempted on Railway and Render; both platforms' free tiers (512MB RAM) are insufficient for the embedding model's memory footprint, causing the process to be killed on startup. Hugging Face Spaces offers a free tier with adequate RAM (16GB) but currently requires a verified payment method even for free-tier usage.

**Current status:** Running locally via Docker. Next steps for production deployment: either verify on Hugging Face Spaces, or reduce memory usage (a smaller embedding model, lazy-loading the model only when needed).

## Design Decisions

- **Why Groq over Gemini:** Initially built on the Gemini API, but its free tier proved too restrictive for iterative testing (5 requests/minute, 20/day). Switched to Groq for a significantly higher free-tier request limit.
- **Why ChromaDB:** A lightweight, easy-to-run vector database suitable for a project of this scale, with no external hosting dependency.
- **Chunking strategy:** Fixed-size word chunks with overlap, to preserve context across chunk boundaries.
- **Error handling:** Retry logic wraps LLM calls to handle transient rate-limit/availability errors gracefully rather than failing the whole request.

## Limitations

- Uploaded PDFs and the vector database are not persisted across container restarts (no external storage attached yet).
- No authentication; any user of a deployed instance could use the shared API quota.
- Tested primarily on English-language, text-based PDFs (not scanned/image-only documents).

## Future Scope

- Persistent storage for uploaded documents and embeddings
- User accounts and per-user document isolation
- Multi-document comparison and cross-referencing
- Quiz/flashcard generation from uploaded notes
- Production deployment once infrastructure constraints are resolved