# Error Fix Documentation

This file documents the fixes applied to resolve the `/chat` endpoint failure in `app.py`.

---

## Error seen

When sending a chat request from the UI (or via API), the server returned **HTTP 500** with:

```json
{
  "error": "Error code: 400 - {'error': {'message': 'code=400, message=tool `HNSW Retriver` cannot contain whitespace in its name, type=invalid_request_error', 'type': 'invalid_request_error'}}"
}
```

The frontend also expected a JSON field named `answer`, but the backend returned `response`, so replies appeared as:

> I didn't get a usable response.

---

## Fix 1: Groq tool names cannot contain spaces

Groq rejects agent tool names that include whitespace. The retriever tools were created with names like `"HNSW Retriver"`.

### Before

```python
brainer = create_retriever_tool(
    retriever,
    name="HNSW Retriver",
    description="Searches internal documents. Use this as your primary source for internal info."
)
```

### After

```python
brainer = create_retriever_tool(
    retriever,
    name="HNSW_Retriever",
    description="Searches internal documents. Use this as your primary source for internal info."
)
```

The same change was applied to all retriever tools:

| Before | After |
|--------|-------|
| `"HNSW Retriver"` | `"HNSW_Retriever"` |
| `"PQ_faiss Retriever"` | `"PQ_faiss_Retriever"` |
| `"IVF_faiss Retriever"` | `"IVF_faiss_Retriever"` |
| `"IVFPQ_faiss Retriever"` | `"IVFPQ_faiss_Retriever"` |

---

## Fix 2: Agent invoke must receive a message list

The agent was invoked with a raw string instead of a LangChain message object list.

### Before

```python
return agent.invoke({'messages': query})
```

### After

```python
from langchain_core.messages import HumanMessage

return agent.invoke({"messages": [HumanMessage(content=query)]})
```

---

## Fix 3: Match frontend JSON response shape

`templates/index.html` reads `data.answer` and `data.sources`. The backend was returning `response` only.

### Before

```python
answer = result['messages'][-1].content
return jsonify({"response": answer})
```

### After

```python
answer = result["messages"][-1].content
return jsonify({"answer": answer, "sources": sources})
```

Support for all UI vector-store options was also added:

- `faiss-hnsw`
- `faiss-pq`
- `faiss-ivf`
- `faiss-ivfpq`
- `pinecone`

---

## How to verify

1. Start the app:

   ```bash
   python app.py
   ```

2. Open `http://127.0.0.1:5000/` and send a message.

3. Or test with curl / PowerShell:

   ```powershell
   Invoke-RestMethod -Uri "http://127.0.0.1:5000/chat" -Method POST -ContentType "application/json" -Body '{"message":"What is Python?","vector_store":"faiss-hnsw"}'
   ```

Expected response shape:

```json
{
  "answer": "...assistant reply...",
  "sources": []
}
```
