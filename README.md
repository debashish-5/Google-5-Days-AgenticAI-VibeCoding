<div align="center">

# 🚀 Google 5 Days VibeCodes
### *The Ultimate 0-to-100% RAG Mastery & Production Agentic AI Teaching Assistant*

![Ongoing Project](https://img.shields.io/badge/Project_Status-Actively_Evolving_🚀-brightgreen?style=for-the-badge)
![RAG Journey](https://img.shields.io/badge/RAG_Level-Beginner_to_Deep_RAG_Hero-FF6F00?style=for-the-badge)
![Vibe Code](https://img.shields.io/badge/100%25-Vibe_Coded_⚡-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

<p align="center">
  <a href="#-tech-stack--skills-ecosystem"><b>Tech Stack</b></a> •
  <a href="#-architecture--pipeline"><b>Architecture</b></a> •
  <a href="#-rag-learning-roadmap-0--100"><b>RAG Roadmap</b></a> •
  <a href="#-index-type-cheatsheet"><b>FAISS Comparison</b></a> •
  <a href="#-quick-start"><b>Quick Start</b></a> •
  <a href="#-project-structure"><b>Project Structure</b></a>
</p>

---

</div>

## 📌 Overview & Vision

**Google 5 Days VibeCodes** is a research-grade, production-aware AI Teaching Assistant framework built to transition learners from **RAG beginners** to **Deep RAG & Agentic AI Practitioners**.

This repo serves a dual purpose:
1. **🧪 Live Code & Learning Lab**: Interactive notebooks and modular backends detailing step-by-step vector search mechanics, query rewriting, hybrid retrieval, and agent tool calling.
2. **🛡️ Production Blueprint**: Dual Flask backends demonstrating local document ingestion, vector quantization, API failure resilience, and observability via **LangSmith**.

---

## 🛠️ Tech Stack & Skills Ecosystem

<div align="center">

| Domain | Technologies & Frameworks |
| :--- | :--- |
| **Backend & Web** | ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white) ![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| **Orchestration** | ![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white) |
| **Vector Engines** | ![FAISS](https://img.shields.io/badge/Meta_FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white) ![Pinecone](https://img.shields.io/badge/Pinecone_DB-000000?style=for-the-badge&logo=pinecone&logoColor=white) |
| **LLM Inference** | ![Groq](https://img.shields.io/badge/Groq_LPU-F05023?style=for-the-badge&logo=fastapi&logoColor=white) |
| **Embeddings** | ![HuggingFace](https://img.shields.io/badge/HuggingFace_MiniLM-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black) |
| **Search & Web AI** | ![Tavily](https://img.shields.io/badge/Tavily_Search_API-4B0082?style=for-the-badge&logo=google-chrome&logoColor=white) |
| **Observability** | ![LangSmith](https://img.shields.io/badge/LangSmith_Tracing-1C1C1C?style=for-the-badge&logo=datadog&logoColor=white) |

</div>

---

## 📐 Architecture & Pipeline

```mermaid
flowchart TD
    subgraph Data Prep & Ingestion
        A[📄 Source PDFs in documents/] --> B[🔪 RecursiveCharacterTextSplitter]
        B --> C[🧠 HuggingFace Embeddings: MiniLM-L6-v2]
        C --> D[(⚡ FAISS Vector Stores: HNSW / PQ / IVF / IVFPQ)]
    end

    subgraph Query Execution & Agent Routing
        E[👤 User Query via Flask UI] --> F{🧠 Query Rewriter / Agent Router}
        F -->|Dense + Sparse| G[🔍 Hybrid Retriever: BM25 + FAISS]
        F -->|Cloud Fallback| H[🌲 Pinecone Vector Store]
        F -->|External Knowledge| I[🌐 Tavily Web Search Tool]
    end

    subgraph Response Generation
        G --> J[📝 Context Aggregator]
        H --> J
        I --> J
        J --> K{Groq API Online?}
        K -->|Yes| L[🤖 ChatGroq Generator]
        K -->|No| M[🛡️ Local Snippet Fallback Engine]
        L --> N[💬 RAG Answer + Source Citations]
        M --> N
    end
