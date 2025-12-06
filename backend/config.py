import os
from dotenv import load_dotenv

load_dotenv()

# Where Chroma stores the index
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/index")

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
