import requests
from typing import List
from .config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, OLLAMA_CHAT_MODEL

def embed_texts_ollama(texts: List[str]) -> List[list]:
    """
    Call Ollama /api/embed to get embeddings for a list of texts.
    """
    url = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "input": texts,
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()

    # Ollama returns: { "embeddings": [[...], [...], ...] }
    return data["embeddings"]

def chat_ollama(system_prompt: str, user_prompt: str) -> str:
    """
    Call Ollama /api/chat for a non-streaming response.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()

    # Response format: { "message": { "role": "...", "content": "..." }, ... }
    return data["message"]["content"]
