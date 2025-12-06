import argparse
from .ingest import ingest_directory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a directory with Ollama embeddings")
    parser.add_argument("path", help="Root directory to index")
    parser.add_argument("--collection", default="default", help="Collection name")
    args = parser.parse_args()

    ingest_directory(args.path, args.collection)
