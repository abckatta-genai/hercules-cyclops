"""
chroma_client.py — Hercules Cyclops, Stage 4
ChromaDB initialisation and collection management.

WHAT THIS FILE DOES:
  - Creates (or loads) a persistent ChromaDB collection
  - Uses cosine similarity — correct for sentence-transformer embeddings
  - Provides get_collection() used by vector_ingestor.py and vector_retriever.py

WHY CHROMADB (local, not Pinecone/Weaviate):
  Runs entirely in-process — no server, no API key, no network calls.
  For an assignment that needs to be cloneable and runnable in under
  10 minutes, a local DB is the right call. Production would use
  Pinecone or Weaviate for scale.

WHY COSINE SIMILARITY:
  sentence-transformers produce unit-normalised vectors. Cosine similarity
  measures the angle between vectors — perfect for semantic similarity.
  Dot product would also work on normalised vectors, but cosine is the
  standard for this use case.

WHY PERSIST_DIRECTORY:
  Without persist_directory, ChromaDB stores everything in memory and
  loses it on restart. Setting persist_directory writes to disk so the
  collection survives between runs — critical for a pipeline where
  ingestion and retrieval happen in separate script runs.

USAGE:
  from src.vector.chroma_client import get_collection
  collection = get_collection()
  # use collection for add/query operations
"""

import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
# ChromaDB writes its SQLite + vector files here.
# Relative to project root — works regardless of where script is run from.
CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma"

# ── Collection config ─────────────────────────────────────────────────────────
COLLECTION_NAME = "hercules_cyclops"


def get_client() -> chromadb.PersistentClient:
    """
    Returns a persistent ChromaDB client.
    Creates CHROMA_DIR if it doesn't exist.
    Data survives restarts — safe to call multiple times.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    log.info(f"ChromaDB client initialised at: {CHROMA_DIR}")
    return client


def get_collection(client: chromadb.PersistentClient = None) -> chromadb.Collection:
    """
    Returns the hercules_cyclops collection.
    Creates it if it doesn't exist (get_or_create is idempotent).

    cosine distance = 1 - cosine_similarity.
    ChromaDB minimises distance, so lower = more similar — correct behaviour.
    """
    if client is None:
        client = get_client()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for embeddings
    )

    log.info(f"Collection '{COLLECTION_NAME}': {collection.count()} vectors")
    return collection


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run directly to verify ChromaDB is working:
      python src/vector/chroma_client.py
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    collection = get_collection()
    print(f"Collection count: {collection.count()} vectors")
    print(f"Chroma dir: {CHROMA_DIR}")
    print("ChromaDB OK")
