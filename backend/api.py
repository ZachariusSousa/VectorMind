from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .ingest import ingest_directory
from .query import answer

app = FastAPI(title="VectorMind Local RAG API")

# CORS so a local React app (localhost:5173, etc.) can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # can be locked later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models ----------

class IngestRequest(BaseModel):
    path: str
    collection: str = "default"


class IngestResponse(BaseModel):
    message: str


class QueryRequest(BaseModel):
    question: str
    collection: str = "default"


class QueryResponse(BaseModel):
    answer: str


# ---------- Endpoints ----------

@app.get("/api/health")
def health_check():
    """
    Simple health endpoint. Later can add real checks
    for Ollama/Chroma/etc.
    """
    return {"status": "ok"}


@app.post("/api/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """
    Runs ingestion on the given path + collection.
    This is synchronous for simplicity: the request will block
    until ingest_directory finishes.
    """
    root = Path(req.path)

    if not root.exists() or not root.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Path does not exist or is not a directory: {root}",
        )

    try:
        ingest_directory(str(root), req.collection)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {e}",
        )

    return IngestResponse(
        message=f"Ingestion completed for {root} into collection '{req.collection}'."
    )


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Ask a question against a collection and return the answer text.
    For now we just wraps existing `answer()` function.
    """
    try:
        resp = answer(req.question, req.collection)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {e}",
        )

    return QueryResponse(answer=resp)
