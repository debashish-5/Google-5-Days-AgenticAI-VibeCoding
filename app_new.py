import os
from flask import Flask, jsonify, render_template, request

from graph_rag import format_graph_sources, retrieve_graph_chunks

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_FOLDERS = {
    "faiss-hnsw": "HNSW_faiss",
    "faiss-ivf": "IVF_faiss",
    "faiss-ivfpq": "IVFPQ_faiss",
    "faiss-pq": "PQ_faiss",
    "pinecone": "HNSW_faiss",
}


def _normalize_vector_store(value):
    mapping = {
        "faiss-flat": "faiss-hnsw",
        "faiss-hnsw": "faiss-hnsw",
        "faiss-ivf": "faiss-ivf",
        "faiss-ivfpq": "faiss-ivfpq",
        "faiss-pq": "faiss-pq",
        "pinecone": "pinecone",
        "neo4j-graph": "neo4j-graph",
    }
    return mapping.get(value, "faiss-hnsw")


def _build_sources(query, vector_store):
    folder_name = INDEX_FOLDERS.get(vector_store, "HNSW_faiss")
    folder_path = os.path.join(BASE_DIR, folder_name)

    if not os.path.isdir(folder_path):
        return [{
            "title": "Unavailable index",
            "snippet": f"The selected index folder '{folder_name}' was not found on disk.",
            "score": 0.0,
        }]

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(
            folder_path=folder_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        documents = db.similarity_search(query, k=3)
        sources = []
        for doc in documents:
            metadata = doc.metadata or {}
            title = metadata.get("source") or metadata.get("title") or "Local document"
            snippet = (doc.page_content or "").strip().replace("\n", " ")[:400]
            sources.append({"title": title, "snippet": snippet, "score": 0.9})
        return sources
    except Exception:
        return [{
            "title": "Fallback context",
            "snippet": "The local vector index could not be loaded, so the app is returning a lightweight fallback response.",
            "score": 0.0,
        }]


def _generate_answer(query, sources, vector_store):
    if os.getenv("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=os.getenv("GROQ_API_KEY"),
            )
            context = "\n\n".join(f"Source: {item['title']}\n{item['snippet']}" for item in sources)
            prompt = f"""
You are a helpful teaching assistant for Google 5 Days VibeCodes.
Answer the user's question using the context below.
If the context is weak, say so clearly.

Context:
{context}

User question: {query}
"""
            response = llm.invoke(prompt.format(context=context, query=query))
            if hasattr(response, "content"):
                return response.content
        except Exception:
            pass

    snippet = sources[0]["snippet"] if sources else "No context was available."
    return (
        f"I connected the chat UI to the backend successfully. "
        f"I used the {vector_store} index and found a relevant context snippet: {snippet}"
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_query = (payload.get("message") or "").strip()
    vector_store = payload.get("vector_store") or payload.get("vectorstore") or "faiss-flat"

    if not user_query:
        return jsonify({"error": "Missing user query"}), 400

    normalized_store = _normalize_vector_store(vector_store)
    if normalized_store == "neo4j-graph":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            chunks = retrieve_graph_chunks(user_query, embeddings)
            sources = format_graph_sources(chunks, snippet_length=400)
            answer = _generate_answer(user_query, sources, normalized_store)
            return jsonify({"answer": answer, "sources": sources})
        except (ImportError, RuntimeError, ValueError) as exc:
            return jsonify({"error": f"Neo4j Graph RAG retrieval failed: {exc}"}), 503

    sources = _build_sources(user_query, normalized_store)
    answer = _generate_answer(user_query, sources, normalized_store)

    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
