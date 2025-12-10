import os
from typing import List, Tuple
from .db import get_collection
from .ollama_client import embed_texts_ollama
from .config import EMBEDDING_BATCH_SIZE

# can add .shader, .uxml, etc. for Unity later
TEXT_EXTS = {
    ".txt", ".md",
    ".py", ".cs", ".js", ".ts",
    ".json", ".yaml", ".yml",
    ".xml", ".shader",
}

def read_text_files(root: str) -> List[Tuple[str, str]]:
    docs = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in TEXT_EXTS:
                path = os.path.join(dirpath, fname)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        if text.strip():
                            docs.append((path, text))
                except Exception as e:
                    print(f"Skipping {path}: {e}")
    return docs

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def ingest_directory(root: str, collection_name: str = "default"):
    coll = get_collection(collection_name)
    docs = read_text_files(root)

    ids: List[str] = []
    contents: List[str] = []
    metadatas: List[dict] = []

    for path, text in docs:
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = f"{path}:{i}"
            ids.append(doc_id)
            contents.append(chunk)
            metadatas.append(
                {
                    "path": path,
                    "chunk": i,
                }
            )

    if not contents:
        print("No files/chunks found to index.")
        return

    total = len(contents)
    print(
        f"Embedding {total} chunks with Ollama "
        f"(batch size={EMBEDDING_BATCH_SIZE})..."
    )

    batch_size = max(1, EMBEDDING_BATCH_SIZE)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_ids = ids[start:end]
        batch_contents = contents[start:end]
        batch_metadatas = metadatas[start:end]

        try:
            embeddings = embed_texts_ollama(batch_contents)
        except Exception as e:
            print(
                f"Error embedding batch {start}-{end} "
                f"({len(batch_contents)} chunks): {e}"
            )
            # Optionally: continue to next batch instead of aborting everything
            raise

        if len(embeddings) != len(batch_contents):
            print(
                f"Warning: embeddings count ({len(embeddings)}) "
                f"!= contents count ({len(batch_contents)}) for batch {start}-{end}"
            )

        coll.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_contents,
            metadatas=batch_metadatas,
        )

        print(f"  Indexed chunks {start + 1}–{end} / {total}")

    print("Done. Index saved.")

