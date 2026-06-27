"""
vector_ingestor.py — Hercules Cyclops, Stage 4
Embeds all extracted items and stores them in ChromaDB.

WHAT THIS FILE DOES:
  Reads every item from data/extracted/*.json, generates a text embedding
  using sentence-transformers all-MiniLM-L6-v2, and stores it in ChromaDB
  alongside metadata for filtered retrieval.

WHY all-MiniLM-L6-v2:
  - Runs locally on CPU — no API calls, no cost, no rate limits
  - 384-dimension embeddings — fast and compact
  - Strong semantic performance for short-to-medium English text
  - Standard benchmark model for this use case

TEXT TO EMBED (per item):
  post    → title + " " + selftext  (full context)
  comment → body                    (comment text only)
  Truncated at 512 tokens (model limit).

METADATA STORED (enables filtered search in vector_retriever.py):
  item_id, type, subreddit, window, author,
  sentiment, sentiment_score, created_utc, permalink

  Example filter: "find semantically similar posts, but only from
  window_1_recent and only from Fitness_India subreddit"
  → ChromaDB applies metadata filter BEFORE vector search

BATCHING:
  Embeddings are generated in batches of EMBED_BATCH_SIZE (default 64).
  ChromaDB upserts are done in batches of CHROMA_BATCH_SIZE (default 100).
  Batching is faster than one-by-one and avoids memory issues.

CHECKPOINTING:
  Tracks ingested IDs in a .checkpoint file.
  Re-running skips already-ingested items safely.
  ChromaDB upsert is also idempotent — safe to re-add existing IDs.

USAGE:
  python src/vector/vector_ingestor.py
"""

import json
import logging
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from chroma_client import get_collection, get_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VECTOR] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
EXTRACTED_DIR   = Path(__file__).resolve().parents[2] / "data" / "extracted"
CHECKPOINT_PATH = EXTRACTED_DIR / "chroma_ingest.checkpoint"

# ── Config ────────────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384 dimensions, fast on CPU, strong semantic performance.
# Downloaded automatically on first run (~90MB), cached in ~/.cache/huggingface/
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64    # items to embed at once — fits comfortably in CPU RAM
CHROMA_BATCH_SIZE = 100  # items to upsert to ChromaDB at once

# Max characters for text — model handles ~512 tokens, ~2000 chars is safe
MAX_TEXT_CHARS = 2000

# All 9 target files
TARGET_FILES = [
    "window_1_recent_Fitness_India",
    "window_1_recent_bangalore",
    "window_1_recent_fitness",
    "window_2_midterm_Fitness_India",
    "window_2_midterm_bangalore",
    "window_2_midterm_fitness",
    "window_3_historical_Fitness_India",
    "window_3_historical_bangalore",
    "window_3_historical_fitness",
]


# ── Text builder ──────────────────────────────────────────────────────────────

def build_text(item: dict) -> str:
    """
    Builds the text to embed for one item.

    Post:    title + selftext — full context gives better embeddings
             than title alone. Selftext often has the actual question/discussion.
    Comment: body only — comments don't have titles.

    key_opinion is appended when available — it's the LLM's one-sentence
    summary of the author's main point, which improves semantic search quality.
    """
    item_type  = item.get("type", "")
    extraction = item.get("extraction", {})
    key_opinion = extraction.get("key_opinion", "")

    if item_type == "post":
        title    = item.get("title", "") or ""
        selftext = item.get("selftext", "") or ""
        text     = f"{title} {selftext}".strip()
    else:
        text = (item.get("body", "") or "").strip()

    # Append key_opinion for richer semantic signal
    if key_opinion and key_opinion not in text:
        text = f"{text} {key_opinion}".strip()

    # Truncate to model limit
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]

    # Fallback for empty text
    if not text:
        text = "[no content]"

    return text


def build_metadata(item: dict) -> dict:
    """
    Builds the metadata dict stored alongside each embedding in ChromaDB.

    ChromaDB metadata values must be str, int, or float — no lists, no None.
    Lists (entities, topics) are joined to comma-separated strings.
    None values are replaced with empty string or 0.

    These fields enable filtered queries in vector_retriever.py:
      collection.query(where={"window": "window_1_recent"}, ...)
    """
    extraction = item.get("extraction", {})

    # Flatten entities list → comma-separated string
    entities = extraction.get("entities", [])
    if entities and isinstance(entities[0], dict):
        entity_names = ", ".join(e.get("name", "") for e in entities if e.get("name"))
    else:
        entity_names = ", ".join(str(e) for e in entities if e)

    # Flatten topics list → comma-separated string
    topics = extraction.get("topics", [])
    topic_str = ", ".join(str(t) for t in topics if t)

    return {
        "item_id":        str(item.get("id", "")),
        "type":           str(item.get("type", "")),
        "subreddit":      str(item.get("subreddit", "")),
        "window":         str(item.get("window", "")),
        "author":         str(item.get("author", "[deleted]")),
        "sentiment":      str(extraction.get("sentiment", "neutral")),
        "sentiment_score": float(extraction.get("sentiment_score", 0.0) or 0.0),
        "created_utc":    int(item.get("created_utc", 0) or 0),
        "permalink":      str(item.get("permalink", "")),
        "entities":       entity_names,
        "topics":         topic_str,
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    if not CHECKPOINT_PATH.exists():
        return set()
    return {line.strip() for line in CHECKPOINT_PATH.read_text().splitlines() if line.strip()}


def save_checkpoint(done_ids: set) -> None:
    CHECKPOINT_PATH.write_text("\n".join(sorted(done_ids)))


# ── Per-file ingestor ─────────────────────────────────────────────────────────

def ingest_file(model: SentenceTransformer, collection, filename: str, done_ids: set) -> int:
    """
    Embeds and ingests all pending items from one extracted file.
    Returns count of newly ingested items.

    Process per batch:
      1. Build text for each item
      2. Batch-embed all texts at once (faster than one-by-one)
      3. Upsert to ChromaDB with IDs, embeddings, documents, metadata
    """
    ext_path = EXTRACTED_DIR / f"{filename}.json"

    if not ext_path.exists():
        log.warning(f"Extracted file not found: {ext_path}")
        return 0

    items = json.load(open(ext_path, encoding="utf-8"))
    log.info(f"\nFILE: {filename} — {len(items)} items")

    pending = [it for it in items if it.get("id") not in done_ids]
    log.info(f"  Pending: {len(pending)} (skipping {len(items)-len(pending)} already done)")

    if not pending:
        log.info("  Nothing to do.")
        return 0

    total_ingested = 0

    # Process in embed batches
    for batch_start in range(0, len(pending), EMBED_BATCH_SIZE):
        batch     = pending[batch_start: batch_start + EMBED_BATCH_SIZE]
        batch_num = batch_start // EMBED_BATCH_SIZE + 1
        total_batches = (len(pending) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

        log.info(f"  Batch {batch_num}/{total_batches} — embedding {len(batch)} items...")

        # Build texts for this batch
        texts = [build_text(item) for item in batch]

        # Embed entire batch at once — much faster than one-by-one
        # show_progress_bar=False keeps logs clean
        embeddings = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
        ).tolist()

        # Build ChromaDB upsert payload
        ids        = [item.get("id") for item in batch]
        metadatas  = [build_metadata(item) for item in batch]
        documents  = texts  # store original text for inspection

        # Upsert to ChromaDB — idempotent, safe to re-run
        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            total_ingested += len(batch)

            # Mark as done
            for item in batch:
                done_ids.add(item.get("id"))
            save_checkpoint(done_ids)

            log.info(f"  Batch {batch_num} done — {len(batch)} upserted | total: {total_ingested}")

        except Exception as e:
            log.error(f"  Batch {batch_num} failed: {e}")
            continue

    return total_ingested


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("Hercules Cyclops — Stage 4: ChromaDB Vector Ingestion")
    log.info(f"Extracted dir: {EXTRACTED_DIR}")
    log.info(f"Embedding model: {EMBEDDING_MODEL}")

    # ── Step 1: load embedding model ──────────────────────────────────────────
    # Downloads ~90MB on first run, cached locally after that.
    log.info("Loading embedding model (downloads on first run, cached after)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    log.info(f"Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")

    # ── Step 2: initialise ChromaDB collection ────────────────────────────────
    client     = get_client()
    collection = get_collection(client)
    log.info(f"ChromaDB collection ready. Current count: {collection.count()}")

    # ── Step 3: load checkpoint ───────────────────────────────────────────────
    done_ids = load_checkpoint()
    log.info(f"Checkpoint: {len(done_ids)} items already ingested")

    # ── Step 4: ingest all files ──────────────────────────────────────────────
    grand_total = 0

    for filename in TARGET_FILES:
        try:
            count = ingest_file(model, collection, filename, done_ids)
            grand_total += count
        except KeyboardInterrupt:
            log.info("\nInterrupted. Progress saved via checkpoint.")
            break
        except Exception as e:
            log.error(f"Fatal error on {filename}: {e}")
            continue

    # ── Step 5: final count ───────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("VECTOR INGESTION COMPLETE")
    log.info(f"Total items ingested this run: {grand_total}")
    log.info(f"Total vectors in ChromaDB:     {collection.count()}")
    log.info("\nStage 4 complete. Proceed to Stage 5 (Retrieval).")


if __name__ == "__main__":
    main()
