"""
vector_retriever.py — Hercules Cyclops, Stage 5
Retrieves semantically similar content from ChromaDB.

WHAT THIS FILE DOES:
  Takes a natural language query, embeds it using the same model used
  during ingestion (all-MiniLM-L6-v2), and runs cosine similarity search
  in ChromaDB to find semantically similar posts/comments.

WHY VECTOR RETRIEVAL (not just graph):
  Vector search captures meaning regardless of exact wording:
  - "feeling unmotivated at the gym" matches posts about "lost the drive
    to lift" even though no keyword overlaps
  - "affordable supplements" matches "budget-friendly protein" semantically
  Graph retrieval would miss these — it only finds exact entity/topic matches.

SAME MODEL AS INGESTION:
  CRITICAL: must use the exact same embedding model used in vector_ingestor.py.
  Different models produce incompatible vector spaces — queries and documents
  must live in the same space for similarity to be meaningful.
  Model: all-MiniLM-L6-v2 (384 dimensions, cosine similarity)

METADATA FILTERING:
  ChromaDB supports pre-filtering by metadata before vector search.
  This is much faster than post-filtering: the ANN search only runs
  on the filtered subset.

  Available filters (from vector_ingestor.py metadata):
    window, subreddit, type, sentiment, author

  Example: find semantically similar posts from window_1_recent only:
    collection.query(where={"window": "window_1_recent"}, ...)

RETURN FORMAT (same structure as graph_retriever.py for RRF compatibility):
  [
    {
      "id":         str,
      "type":       str,
      "text":       str,
      "score":      float,  # cosine similarity 0.0-1.0 (higher = more similar)
      "subreddit":  str,
      "window":     str,
      "author":     str,
      "permalink":  str,
      "sentiment":  str,
      "sentiment_score": float,
      "source":     "vector"
    },
    ...
  ]

USAGE:
  from src.retrieval.vector_retriever import VectorRetriever
  retriever = VectorRetriever()
  results = retriever.retrieve("feeling unmotivated at the gym", top_k=10)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

# Allow imports from src/vector/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vector"))
from chroma_client import get_collection, get_client

log = logging.getLogger(__name__)

# Must match the model used in vector_ingestor.py exactly
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class VectorRetriever:
    """
    Retrieves semantically similar content from ChromaDB.

    Usage:
      retriever = VectorRetriever()
      results = retriever.retrieve("affordable protein powder India", top_k=10)
    """

    def __init__(self):
        log.info("Loading embedding model...")
        self.model      = SentenceTransformer(EMBEDDING_MODEL)
        self.client     = get_client()
        self.collection = get_collection(self.client)
        log.info(f"VectorRetriever ready. Collection has {self.collection.count()} vectors.")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        window: Optional[str] = None,
        windows: Optional[list] = None,
        subreddit: Optional[str] = None,
        sentiment: Optional[str] = None,
    ) -> list[dict]:
        """
        Main retrieval method. Embeds query and runs cosine similarity search.

        Args:
          query:     Natural language query string
          top_k:     Number of results to return
          window:    Filter to single time window e.g. "window_1_recent"
          windows:   Filter to list of windows (for comparison queries)
          subreddit: Filter to specific subreddit e.g. "Fitness_India"
          sentiment: Filter to specific sentiment e.g. "positive"

        Returns:
          List of result dicts sorted by similarity score descending.
        """
        log.info(f"VectorRetriever: query='{query}'")

        # Embed the query using same model as ingestion
        query_embedding = self.model.encode(query).tolist()

        # Build metadata filter (applied BEFORE vector search)
        where_filter = self._build_filter(window, windows, subreddit, sentiment)

        # Run ChromaDB query
        try:
            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results":        min(top_k, self.collection.count()),
                "include":          ["documents", "metadatas", "distances"],
            }

            # Only add where filter if we have one
            # ChromaDB errors if where={} is passed with no conditions
            if where_filter:
                query_kwargs["where"] = where_filter

            results = self.collection.query(**query_kwargs)

        except Exception as e:
            log.error(f"ChromaDB query failed: {e}")
            return []

        # Normalise results to standard format
        return self._normalise_results(results)

    def retrieve_for_comparison(
        self,
        query: str,
        window_a: str,
        window_b: str,
        top_k: int = 10,
    ) -> dict:
        """
        Retrieves results separately for two time windows.
        Used for time-series comparison queries in demo.py.

        Returns:
          {"window_a": [...results...], "window_b": [...results...]}
        """
        results_a = self.retrieve(query, top_k=top_k, window=window_a)
        results_b = self.retrieve(query, top_k=top_k, window=window_b)
        return {"window_a": results_a, "window_b": results_b}

    def _build_filter(
        self,
        window: Optional[str],
        windows: Optional[list],
        subreddit: Optional[str],
        sentiment: Optional[str],
    ) -> Optional[dict]:
        """
        Builds ChromaDB where filter dict.

        ChromaDB filter syntax:
          Single condition:   {"field": {"$eq": "value"}}
          Multiple conditions: {"$and": [{"field": {"$eq": "v1"}}, ...]}
          In list:            {"field": {"$in": ["v1", "v2"]}}
        """
        conditions = []

        if window:
            conditions.append({"window": {"$eq": window}})
        elif windows and len(windows) > 0:
            conditions.append({"window": {"$in": windows}})

        if subreddit:
            conditions.append({"subreddit": {"$eq": subreddit}})

        if sentiment:
            conditions.append({"sentiment": {"$eq": sentiment}})

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def _normalise_results(self, chroma_results: dict) -> list[dict]:
        """
        Converts ChromaDB query output to standard result format.

        ChromaDB returns:
          distances:  list of distances (lower = more similar for cosine)
          metadatas:  list of metadata dicts
          documents:  list of original text strings

        We convert distance to similarity: similarity = 1 - distance
        ChromaDB cosine distance is in [0, 2]; similarity in [-1, 1].
        Practically for good matches: similarity > 0.3
        """
        normalised = []

        ids        = chroma_results.get("ids", [[]])[0]
        distances  = chroma_results.get("distances", [[]])[0]
        metadatas  = chroma_results.get("metadatas", [[]])[0]
        documents  = chroma_results.get("documents", [[]])[0]

        for i, item_id in enumerate(ids):
            distance   = distances[i] if i < len(distances) else 1.0
            metadata   = metadatas[i] if i < len(metadatas) else {}
            document   = documents[i] if i < len(documents) else ""

            # Convert distance to similarity score (higher = better)
            similarity = max(0.0, 1.0 - distance)

            normalised.append({
                "id":              item_id,
                "type":            metadata.get("type", ""),
                "text":            document,
                "score":           similarity,
                "subreddit":       metadata.get("subreddit", ""),
                "window":          metadata.get("window", ""),
                "author":          metadata.get("author", ""),
                "permalink":       metadata.get("permalink", ""),
                "sentiment":       metadata.get("sentiment", "neutral"),
                "sentiment_score": float(metadata.get("sentiment_score", 0.0) or 0.0),
                "source":          "vector",
            })

        # Sort by similarity descending
        normalised.sort(key=lambda x: x["score"], reverse=True)
        log.info(f"  Vector: {len(normalised)} results")
        return normalised


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    retriever = VectorRetriever()

    print("\nTest 1: Semantic search — 'feeling unmotivated at the gym'")
    results = retriever.retrieve("feeling unmotivated at the gym", top_k=5)
    for i, r in enumerate(results):
        print(f"  [{i+1}] score={r['score']:.3f} window={r['window']} "
              f"subreddit={r['subreddit']} text={r['text'][:80]}...")

    print("\nTest 2: Filtered search — 'protein supplement' in window_1_recent")
    results = retriever.retrieve(
        "protein supplement recommendation",
        top_k=5,
        window="window_1_recent",
    )
    for i, r in enumerate(results):
        print(f"  [{i+1}] score={r['score']:.3f} window={r['window']} text={r['text'][:80]}...")

    print(f"\nCollection size: {retriever.collection.count()} vectors")
