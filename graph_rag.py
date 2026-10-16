"""Neo4j vector retrieval with optional graph-neighbor context."""

import os


_VECTOR_SEARCH_QUERY = """
CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
YIELD node, score
OPTIONAL MATCH (node)-[relationship]-(neighbor)
WITH node, score,
     collect(CASE WHEN relationship IS NULL THEN null ELSE {
         type: type(relationship),
         labels: labels(neighbor),
         name: coalesce(neighbor.name, neighbor.title, neighbor.topic, neighbor.text, neighbor.id)
     } END)[..5] AS neighbors
RETURN node.id AS id, node.text AS text, node.topic AS topic,
       score, neighbors
ORDER BY score DESC
"""


def retrieve_graph_chunks(question, embeddings, top_k=5):
    """Retrieve relevant Neo4j chunks and any directly connected graph neighbors."""
    try:
        from neo4j import GraphDatabase
        from neo4j.exceptions import Neo4jError
    except ImportError as exc:
        raise RuntimeError("Install the 'neo4j' package to use Neo4j Graph RAG.") from exc

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError("Neo4j is not configured. Set NEO4J_PASSWORD in your environment.")

    query_vector = embeddings.embed_query(question)
    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        database = os.getenv("NEO4J_DATABASE")
        session_options = {"database": database} if database else {}
        with driver.session(**session_options) as session:
            result = session.run(
                _VECTOR_SEARCH_QUERY,
                index_name=os.getenv("NEO4J_VECTOR_INDEX", "chunk_embedding_index"),
                top_k=top_k,
                query_vector=query_vector,
            )
            return [record.data() for record in result]
    except Neo4jError as exc:
        raise RuntimeError(f"Neo4j retrieval failed: {exc}") from exc
    finally:
        if driver is not None:
            driver.close()


def format_graph_chunk(chunk):
    """Format a retrieved chunk and its direct neighbors as answer context."""
    topic = chunk.get("topic") or chunk.get("id") or "Knowledge graph"
    text = chunk.get("text") or ""
    neighbors = chunk.get("neighbors") or []
    neighbor_context = "\n".join(
        f"- {', '.join(item.get('labels') or [])} "
        f"{item.get('type') or 'RELATED_TO'} {item.get('name') or ''}".strip()
        for item in neighbors
    )
    context = f"Topic: {topic}\n{text}"
    if neighbor_context:
        context += f"\nRelated graph context:\n{neighbor_context}"
    return topic, context


def format_graph_sources(chunks, snippet_length=500):
    """Build the source objects expected by the chat UI."""
    sources = []
    for chunk in chunks:
        topic, context = format_graph_chunk(chunk)
        sources.append({
            "title": f"Neo4j: {topic}",
            "snippet": context[:snippet_length],
            "score": float(chunk.get("score", 0.0)),
        })
    return sources
