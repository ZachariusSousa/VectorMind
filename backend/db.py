import chromadb
from .config import CHROMA_DIR

# Uses a persistent client so data survives between processes
_client = chromadb.PersistentClient(path=CHROMA_DIR)

def get_collection(name: str = "default"):
    return _client.get_or_create_collection(name=name)
