from typing import List
from .db import get_collection
from .ollama_client import embed_texts_ollama, chat_ollama

def embed_query_ollama(query: str) -> List[float]:
    # Reuses same embedding model, just single-item list
    return embed_texts_ollama([query])[0]

def search(query: str, collection_name: str = "default", top_k: int = 6):
    coll = get_collection(collection_name)
    q_emb = embed_query_ollama(query)
    results = coll.query(
        query_embeddings=[q_emb],
        n_results=top_k,
    )
    return results  # dict with "documents", "metadatas", "ids", ...

def answer(query: str, collection_name: str = "default") -> str:
    results = search(query, collection_name)
    if not results["documents"] or not results["documents"][0]:
        return "I couldn't find anything relevant in the indexed files."

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_blocks = []
    for doc, meta in zip(docs, metas):
        context_blocks.append(
            f"File: {meta.get('path')} (chunk {meta.get('chunk')})\n{doc}\n"
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

    return chat_ollama(system_prompt, user_prompt).strip()
