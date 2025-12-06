# VectorMind

VectorMind is a local, privacy-preserving project assistant designed to provide context-aware answers about any codebase or directory on your machine. It automatically indexes, chunks, embeds, and stores project files, then uses Retrieval-Augmented Generation (RAG) with a local LLM to deliver accurate and grounded responses.

The goal of VectorMind is to serve as a universal development assistant for tasks such as debugging, code review, architecture insight, and game development support (Unity, Python, C#, or general engine workflows).

## Features

- Local directory indexing with recursive file crawling
- Chunking and embedding of source code, documentation, and configuration files
- Local vector database (ChromaDB) for fast similarity search
- Query engine that retrieves relevant project context for each question
- Integration with local LLMs through Ollama for grounded, project-aware responses
- FastAPI backend providing REST endpoints for indexing and querying
- Optional React frontend for interactive use

## Tech Stack

- Backend: Python, FastAPI  
- Vector Store: ChromaDB  
- Embeddings: BGE-base or BGE-M3 (local via SentenceTransformers)  
- Local LLM Runtime: Ollama (Llama 3.2, CodeLlama, Qwen2.5-Coder, etc.)  
- Frontend: React (optional for MVP)

