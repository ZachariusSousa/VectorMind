from typing import List, Dict, Any
from .db import get_collection
from .ollama_client import embed_texts_ollama, chat_ollama, OllamaError


def embed_query_ollama(query: str) -> List[float]:
    embeddings = embed_texts_ollama([query])
    if not embeddings:
        raise OllamaError("Failed to embed query: empty embeddings list.")
    return embeddings[0]


def search(
    query: str,
    collection_name: str = "default",
    top_k: int = 6,
) -> Dict[str, Any]:
    coll = get_collection(collection_name)

    try:
        q_emb = embed_query_ollama(query)
    except Exception as e:
        raise RuntimeError(f"Error embedding query: {e}")

    try:
        results = coll.query(
            query_embeddings=[q_emb],
            n_results=top_k,
        )
    except Exception as e:
        raise RuntimeError(f"Error querying Chroma collection '{collection_name}': {e}")

    # Basic shape validation
    if not isinstance(results, dict):
        raise RuntimeError(
            f"Unexpected result type from Chroma: {type(results).__name__}"
        )

    return results  # dict with "documents", "metadatas", "ids", ...


def answer(query: str, collection_name: str = "default") -> str:
    try:
        results = search(query, collection_name)
    except Exception as e:
        return f"Sorry, I ran into an error while searching the index: {e}"

    docs_lists = results.get("documents") or []
    metas_lists = results.get("metadatas") or []

    if not docs_lists or not docs_lists[0]:
        return "I couldn't find anything relevant in the indexed files."

    docs = docs_lists[0]
    metas = metas_lists[0] if metas_lists else [{}] * len(docs)

    context_blocks = []
    for doc, meta in zip(docs, metas):
        path = meta.get("path", "unknown file")
        chunk_idx = meta.get("chunk", "?")
        context_blocks.append(
            f"File: {path} (chunk {chunk_idx})\n{doc}\n"
        )
    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant that answers questions about a local codebase and files. "
        "Use only the provided context. When useful, mention file paths and approximate locations."
    )

    user_prompt = f"""
User question:
{query}

Relevant file excerpts:
{context}

Using ONLY the information above, answer the question concisely.
If the answer isn't clear from the context, say you are unsure rather than guessing.
"""

    try:
        return chat_ollama(system_prompt, user_prompt).strip()
    except OllamaError as e:
        return f"Sorry, I ran into an error while calling Ollama: {e}"
    except Exception as e:
        return f"Sorry, something unexpected went wrong while generating the answer: {e}"
