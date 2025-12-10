import time
from typing import List
import requests
from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_REQUEST_TIMEOUT,
    OLLAMA_MAX_RETRIES,
)

class OllamaError(RuntimeError):
    pass


def _post_with_retries(url: str, payload: dict) -> dict:
    """
    Helper: POST with timeout + retries + basic exponential backoff.
    Raises OllamaError if all retries fail.
    """
    last_exc = None
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=OLLAMA_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            # Simple backoff: 1s, 2s, 4s, ...
            sleep_time = min(2 ** (attempt - 1), 10)
            print(
                f"[Ollama] Request failed (attempt {attempt}/{OLLAMA_MAX_RETRIES}): "
                f"{exc}. Retrying in {sleep_time}s..."
            )
            time.sleep(sleep_time)

    raise OllamaError(f"Ollama request failed after {OLLAMA_MAX_RETRIES} attempts: {last_exc}")


def embed_texts_ollama(texts: List[str]) -> List[list]:
    """
    Call Ollama /api/embed to get embeddings for a list of texts.
    """
    if not texts:
        return []

    url = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "input": texts,
    }

    data = _post_with_retries(url, payload)

    # Ollama returns: { "embeddings": [[...], [...], ...] }
    embeddings = data.get("embeddings")
    if embeddings is None or not isinstance(embeddings, list):
        raise OllamaError(
            f"Unexpected response from Ollama /api/embed: missing or invalid 'embeddings' field"
        )

    if len(embeddings) != len(texts):
        print(
            f"[Ollama] Warning: number of embeddings ({len(embeddings)}) "
            f"does not match number of inputs ({len(texts)})"
        )

    return embeddings


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

    data = _post_with_retries(url, payload)

    # Expected format: { "message": { "role": "...", "content": "..." }, ... }
    msg = data.get("message")
    if not isinstance(msg, dict):
        raise OllamaError(
            f"Unexpected response from Ollama /api/chat: missing or invalid 'message' field"
        )

    content = msg.get("content")
    if not isinstance(content, str):
        raise OllamaError(
            f"Unexpected response from Ollama /api/chat: missing or invalid 'content' field"
        )

    return content
