# Google 5 Days VibeCodes


A production-aware AI teaching assistant prototype that combines Flask, local FAISS vector search, hybrid retrieval, PDF document ingestion, and Groq-backed conversational generation.

## About this project

Google 5 Days VibeCodes is a research-grade proof of concept for a local AI assistant. It is built to explore how document embeddings, vector retrieval, and conversational LLM responses can be combined into a lightweight teaching assistant experience.

This repository includes two complementary Flask backends:
- `app.py` — a feature-rich implementation with agent tools, hybrid search, Pinecone support, Tavily web search, and conversational retrievers.
- `app_new.py` — a cleaner and more stable endpoint for selecting local FAISS indexes and generating answers with Groq.

The project is designed to help learners progress from RAG beginner to advanced practitioner, with daily notes and insights added as the code evolves.

The project demonstrates:
- local PDF ingestion and chunking
- FAISS vector search across multiple index types
- hybrid retrieval using BM25 + FAISS
- agent-based search with tool-enabled retrieval
- fallback response generation when external APIs are unavailable

## Why this repo matters

This codebase is designed for learning and experimentation. It showcases practical steps for moving from raw document data to vector search and conversational QA, while also exposing production improvement opportunities such as dependency management, environment configuration, error handling, and deployment hardening.

## Key features

- Local document ingestion from `documents/`
- PDF chunking via `RecursiveCharacterTextSplitter`
- Embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- FAISS index loading for: `HNSW`, `PQ`, `IVF`, `IVFPQ`
- Hybrid retrieval with `BM25Retriever` + `FAISS`
- Pinecone retrieval support via `PineconeVectorStore`
- Groq LLM integration with `ChatGroq`
- Tavily web search tool integration for external knowledge
- Multiple retrieval strategies including agent-driven retrievers
- Lightweight front-end UI in `templates/index.html`

## Architecture

1. `documents/` contains source PDF files.
2. PDF text is loaded and split into chunks.
3. Chunks are embedded using a HuggingFace sentence-transformer model.
4. Local FAISS indexes are built or loaded from disk.
5. User queries are routed to a vector retriever, optionally combined with BM25.
6. Retrieved context is passed to a Groq LLM or fallback generator.
7. The Flask UI at `/` sends requests to `/chat` and renders the answer with sources.

## Project structure

- `app.py` — full prototype with hybrid retrievers, Pinecone, Tavily, and multiple local index formats.
- `app_new.py` — simplified backend with index normalization and fallback response support.
- `templates/index.html` — browser-based chat interface.
- `documents/` — PDF sources.
- `HNSW_faiss/`, `PQ_faiss/`, `IVF_faiss/`, `IVFPQ_faiss/` — prebuilt local vector indexes.
- `*.ipynb` — notebooks for exploration, indexing, and testing.

## Setup

### 1. Create a virtual environment

```powershell
cd "c:\Users\ddeba\Desktop\github project\Google 5 Days VibeCodes"
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install flask python-dotenv langchain-core langchain-groq langchain-classic langchain-community langchain-text-splitters requests langsmith pinecone-client
```

### 3. Add environment variables

Create a `.env` file with keys for optional integrations:

```text
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
TAVILY_API_KEY=your_tavily_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
```

### 4. Prepare documents and indexes

- Place source PDFs into `documents/`
- Ensure the FAISS folders exist before running the app
- Rebuild indexes if needed with a dedicated index generation script or notebook

## Running the app

Use either backend entrypoint:

```powershell
python app_new.py
```

or

```powershell
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

## Usage notes

- `app_new.py` normalizes requested index names and loads the corresponding FAISS folder.
- The `/chat` endpoint accepts JSON with `message` and `vector_store` values.
- `app.py` is more experimental and exposes agent-style retrievers, Tavily search, and Pinecone integration.
- When `GROQ_API_KEY` is missing or invalid, `app_new.py` falls back to a local snippet-based answer.

## Production improvements added

This README now includes production-aware recommendations:
- add a pinned dependency manifest such as `requirements.txt`
- use explicit environment configuration and secret management
- separate development and production settings
- add robust index existence checks and clearer runtime errors
- provide Docker and Procfile deployment support
- add health checks, logging, and runtime monitoring
- implement fallback behavior for API failures
- avoid loading unsafe serialized data in untrusted environments

## New notes and knowledge

- `app_new.py` is the most stable entrypoint for local FAISS retrieval.
- `app.py` contains advanced retrieval experiments, but it is not yet hardened for production.
- Keep external API keys out of source control; use `.env` and secret storage.
- Rebuilding FAISS indexes should be a separate preprocessing step rather than an on-demand runtime operation.
- Use lightweight local embeddings for prototyping, and switch to managed embeddings or cached vectors for production scale.
- If Groq is unavailable, preserve user experience with a clear fallback answer and context snippet.

## Recommendations for next steps

- add `requirements.txt` and pin dependency versions
- create `Dockerfile` and `docker-compose.yml`
- implement a build step to generate or refresh FAISS indexes
- add a dedicated `data/` or `cache/` folder for generated vectors
- improve error handling around missing index folders and invalid user input
- add automated tests for `/chat` and retrieval logic
- separate the indexing pipeline from the web request pipeline

## License

This repository is a proof-of-concept research project. Adapt and extend it with care for any production use.
