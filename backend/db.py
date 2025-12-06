import chromadb
from chromadb.config import Settings
from .config import CHROMA_DIR

_client = chromadb.Client(
    Settings(
        anonymized_telemetry=False,
        persist_directory=CHROMA_DIR,
    )
)

def get_collection(name: str = "default"):
    return _client.get_or_create_collection(name=name)
