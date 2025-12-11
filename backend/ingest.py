import os
import hashlib
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple sliding-window chunker with overlap, in characters.
    """
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


def _is_text_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in TEXT_EXTS


def _iter_text_paths(root: str) -> List[str]:
    """
    Walk the directory tree and collect paths of files with text-like extensions.
    """
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            if _is_text_file(full_path):
                paths.append(full_path)
    return paths


def _load_and_prepare_file(
    path: str,
    old_hash: str | None,
    max_chars: int = 1200,
    overlap: int = 200,
) -> Dict[str, Any] | None:
    """
    Worker for ThreadPoolExecutor:
    - Reads the file
    - Computes SHA-256 hash
    - If unchanged vs old_hash -> return {"changed": False, "path": ...}
    - If changed/new -> chunk text and return chunk data

    Returns:
        None if file couldn't be read or is empty;
        dict with keys:
            "changed": bool
            "path": str
            "file_hash": str (if changed)
            "chunks": List[Tuple[int, str]] (index, chunk_text) (if changed)
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        print(f"Skipping {path}: error reading file: {e}")
        return None

    if not text.strip():
        # Empty or whitespace-only file, skip
        return None

    # Compute content hash
    file_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    if old_hash is not None and file_hash == old_hash:
        # File unchanged; no need to re-chunk / re-embed
        return {
            "changed": False,
            "path": path,
        }

    chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
    indexed_chunks = list(enumerate(chunks))  # (chunk_idx, chunk_text)

    return {
        "changed": True,
        "path": path,
        "file_hash": file_hash,
        "chunks": indexed_chunks,
    }


def ingest_directory(root: str, collection_name: str = "default"):
    """
    Incremental, batched, parallel ingestion:

    1. Load existing docs from Chroma (to get file hashes + ids).
    2. Walk filesystem to collect current text files.
    3. Detect deleted files and remove their chunks from the index.
    4. In parallel, read + hash + chunk files:
        - If hash unchanged -> skip re-embedding.
        - If changed/new -> delete old chunks (if any), then re-embed and add.
    5. Embed new/changed chunks in batches and add to Chroma.
    """
    coll = get_collection(collection_name)

    # --- Step 1: inspect existing index for this collection ---
    print(f"Loading existing index metadata from collection '{collection_name}'...")
    try:
        existing = coll.get(include=["metadatas"])
    except Exception as e:
        print(f"Warning: failed to load existing collection data: {e}")
        existing = {"ids": [], "metadatas": []}

    existing_ids = existing.get("ids") or []
    existing_metas = existing.get("metadatas") or []

    from collections import defaultdict

    existing_hash_by_path: Dict[str, str] = {}
    existing_ids_by_path: Dict[str, List[str]] = defaultdict(list)

    for doc_id, meta in zip(existing_ids, existing_metas):
        if not isinstance(meta, dict):
            continue
        path = meta.get("path")
        if not path:
            continue

        existing_ids_by_path[path].append(doc_id)
        file_hash = meta.get("file_hash")
        if file_hash and path not in existing_hash_by_path:
            existing_hash_by_path[path] = file_hash

    # --- Step 2: collect current filesystem paths ---
    print(f"Scanning '{root}' for text files...")
    current_paths = _iter_text_paths(root)
    current_paths_set = set(current_paths)
    print(f"Found {len(current_paths)} text-like files.")

    # --- Step 3: detect deleted files and remove from index ---
    indexed_paths_set = set(existing_ids_by_path.keys())
    deleted_paths = indexed_paths_set - current_paths_set

    if deleted_paths:
        print(f"Removing {len(deleted_paths)} deleted files from index...")
        for path in deleted_paths:
            ids_to_delete = existing_ids_by_path.get(path) or []
            if not ids_to_delete:
                continue
            try:
                coll.delete(ids=ids_to_delete)
                print(f"  Removed {len(ids_to_delete)} chunks for deleted file: {path}")
            except Exception as e:
                print(
                    f"  Warning: failed to delete chunks for {path}: {e}"
                )
    else:
        print("No deleted files detected in index.")

    # --- Step 4: parallel read + hash + chunk for current files ---
    if not current_paths:
        print("No files/chunks found to index.")
        return

    print("Reading, hashing, and chunking files in parallel...")
    max_workers = max(1, (os.cpu_count() or 4))
    print(f"Using up to {max_workers} worker threads.")

    # Global buffers for new/changed chunks
    ids: List[str] = []
    contents: List[str] = []
    metadatas: List[dict] = []

    changed_paths: set[str] = set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for path in current_paths:
            old_hash = existing_hash_by_path.get(path)
            fut = executor.submit(_load_and_prepare_file, path, old_hash)
            futures[fut] = path

        for fut in as_completed(futures):
            path = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"Error processing file {path}: {e}")
                continue

            if result is None:
                # unreadable or empty file
                continue

            if not result.get("changed"):
                # unchanged file, nothing to add
                continue

            # Mark path as changed so we can delete old chunks
            changed_paths.add(path)

            file_hash = result["file_hash"]
            for chunk_idx, chunk_text in result["chunks"]:
                doc_id = f"{path}:{chunk_idx}"
                ids.append(doc_id)
                contents.append(chunk_text)
                metadatas.append(
                    {
                        "path": path,
                        "chunk": chunk_idx,
                        "file_hash": file_hash,
                    }
                )

    # --- Step 5: delete old chunks for changed files ---
    if changed_paths:
        print(f"{len(changed_paths)} files changed or new; updating index...")
        for path in changed_paths:
            old_ids = existing_ids_by_path.get(path) or []
            if not old_ids:
                continue
            try:
                coll.delete(ids=old_ids)
                print(f"  Removed {len(old_ids)} old chunks for changed file: {path}")
            except Exception as e:
                print(
                    f"  Warning: failed to delete old chunks for changed file {path}: {e}"
                )
    else:
        print("No files changed since last ingest; index is up to date.")
        # If no new/changed chunks, we can return early.
        if not contents:
            return

    # --- Step 6: embed new/changed chunks in batches + add to Chroma ---
    total = len(contents)
    if total == 0:
        print("No new or changed chunks to embed.")
        return

    batch_size = max(1, EMBEDDING_BATCH_SIZE)
    print(
        f"Embedding {total} new/changed chunks with Ollama "
        f"(batch size={batch_size})..."
    )

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
            # continue instead of abborting entire ingest
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
