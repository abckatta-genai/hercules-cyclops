"""
graph_ingestor.py — Hercules Cyclops, Stage 3
Reads all extracted JSON files and ingests them into Neo4j as a property graph.

WHAT THIS FILE DOES:
  Reads every item from data/extracted/*.json and creates:

  NODES:
    (Post)       — one per Reddit post
    (Comment)    — one per Reddit comment
    (Author)     — one per unique username (deduplicated via MERGE)
    (Entity)     — one per unique (name, type) pair e.g. ("MuscleBlaze", "Brand")
    (Topic)      — one per unique topic string e.g. "protein powder"
    (Subreddit)  — one per unique subreddit name

  RELATIONSHIPS:
    (Author)    -[:POSTED]->     (Post)       author wrote this post
    (Author)    -[:COMMENTED]->  (Comment)    author wrote this comment
    (Comment)   -[:ON]->         (Post)       comment belongs to this post
    (Post)      -[:MENTIONS]->   (Entity)     post mentions this entity
    (Comment)   -[:MENTIONS]->   (Entity)     comment mentions this entity
    (Post)      -[:TAGGED]->     (Topic)      post is tagged with this topic
    (Post)      -[:IN]->         (Subreddit)  post is in this subreddit
    (Comment)   -[:IN]->         (Subreddit)  comment is in this subreddit

WHY MERGE (not CREATE):
  MERGE = "find this node if it exists, create it if not."
  The same author appears across hundreds of posts — MERGE gives us ONE
  Author node with hundreds of POSTED relationships, not hundreds of
  duplicate Author nodes. Same for Entity and Topic nodes.

BATCHING:
  Items are processed in batches of BATCH_SIZE (default 100).
  Each batch is one transaction. If a batch fails, only that batch
  is lost — not the entire ingestion run.

CHECKPOINTING:
  A .checkpoint file tracks ingested item IDs. Re-running the script
  skips already-ingested items safely.

USAGE:
  export $(grep NEO4J .env | xargs)
  python src/graph/graph_ingestor.py
"""

import json
import logging
import os
import time
from pathlib import Path
from neo4j import GraphDatabase

from neo4j_client import get_driver, close_driver, run_schema_setup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INGESTOR] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
EXTRACTED_DIR = Path(__file__).resolve().parents[2] / "data" / "extracted"
CHECKPOINT_PATH = EXTRACTED_DIR / "neo4j_ingest.checkpoint"

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 100   # items per transaction — balances memory vs commit frequency

# All 9 extracted files to ingest
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


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    """Returns set of already-ingested item IDs."""
    if not CHECKPOINT_PATH.exists():
        return set()
    return {line.strip() for line in CHECKPOINT_PATH.read_text().splitlines() if line.strip()}


def save_checkpoint(done_ids: set) -> None:
    """Saves ingested item IDs to checkpoint file."""
    CHECKPOINT_PATH.write_text("\n".join(sorted(done_ids)))


# ── Cypher queries ────────────────────────────────────────────────────────────
# Each query uses parameterised values ($param) — never string formatting.
# This prevents injection and lets Neo4j cache the query plan.

MERGE_POST = """
MERGE (p:Post {id: $id})
SET p.title        = $title,
    p.selftext     = $selftext,
    p.score        = $score,
    p.num_comments = $num_comments,
    p.created_utc  = $created_utc,
    p.window       = $window,
    p.subreddit    = $subreddit,
    p.permalink    = $permalink,
    p.sentiment    = $sentiment,
    p.sentiment_score = $sentiment_score,
    p.key_opinion  = $key_opinion

MERGE (a:Author {name: $author})
MERGE (s:Subreddit {name: $subreddit})

MERGE (a)-[:POSTED]->(p)
MERGE (p)-[:IN]->(s)
"""

MERGE_COMMENT = """
MERGE (c:Comment {id: $id})
SET c.body          = $body,
    c.score         = $score,
    c.created_utc   = $created_utc,
    c.window        = $window,
    c.subreddit     = $subreddit,
    c.permalink     = $permalink,
    c.parent_id     = $parent_id,
    c.post_id       = $post_id,
    c.sentiment     = $sentiment,
    c.sentiment_score = $sentiment_score,
    c.key_opinion   = $key_opinion

MERGE (a:Author {name: $author})
MERGE (s:Subreddit {name: $subreddit})

MERGE (a)-[:COMMENTED]->(c)
MERGE (c)-[:IN]->(s)
"""

MERGE_COMMENT_ON_POST = """
MATCH (c:Comment {id: $comment_id})
MATCH (p:Post {id: $post_id})
MERGE (c)-[:ON]->(p)
"""

MERGE_ENTITY = """
MATCH (n {id: $item_id})
MERGE (e:Entity {name: $name, type: $entity_type})
MERGE (n)-[:MENTIONS]->(e)
"""

MERGE_TOPIC = """
MATCH (p:Post {id: $item_id})
MERGE (t:Topic {name: $topic})
MERGE (p)-[:TAGGED]->(t)
"""


# ── Per-item ingestion ────────────────────────────────────────────────────────

def ingest_item(session, item: dict) -> None:
    """
    Ingests one extracted item (post or comment) into Neo4j.
    Creates all nodes and relationships for that item in one session.

    Extraction fields are accessed safely with .get() + fallbacks
    so missing/None values never break the Cypher query.
    """
    extraction    = item.get("extraction", {})
    item_id       = item.get("id")
    item_type     = item.get("type")
    sentiment     = extraction.get("sentiment", "neutral")
    sentiment_score = extraction.get("sentiment_score", 0.0)
    key_opinion   = extraction.get("key_opinion", "")
    entities      = extraction.get("entities", [])
    topics        = extraction.get("topics", [])

    if item_type == "post":
        # ── Create Post node + Author node + Subreddit node + edges ──────────
        session.run(MERGE_POST, {
            "id":            item_id,
            "title":         item.get("title", ""),
            "selftext":      item.get("selftext", ""),
            "score":         item.get("score", 0),
            "num_comments":  item.get("num_comments", 0),
            "created_utc":   item.get("created_utc", 0),
            "window":        item.get("window", "unknown"),
            "subreddit":     item.get("subreddit", ""),
            "permalink":     item.get("permalink", ""),
            "author":        item.get("author", "[deleted]"),
            "sentiment":     sentiment,
            "sentiment_score": float(sentiment_score) if sentiment_score is not None else 0.0,
            "key_opinion":   key_opinion,
        })

        # ── Tag topics (Post only — topics are post-level) ────────────────────
        for topic in topics:
            if topic and isinstance(topic, str):
                session.run(MERGE_TOPIC, {
                    "item_id": item_id,
                    "topic":   topic.strip().lower(),  # normalise case
                })

    elif item_type == "comment":
        # ── Create Comment node + Author node + Subreddit node + edges ───────
        session.run(MERGE_COMMENT, {
            "id":            item_id,
            "body":          item.get("body", ""),
            "score":         item.get("score", 0),
            "created_utc":   item.get("created_utc", 0),
            "window":        item.get("window", "unknown"),
            "subreddit":     item.get("subreddit", ""),
            "permalink":     item.get("permalink", ""),
            "author":        item.get("author", "[deleted]"),
            "parent_id":     item.get("parent_id", ""),
            "post_id":       item.get("post_id", ""),
            "sentiment":     sentiment,
            "sentiment_score": float(sentiment_score) if sentiment_score is not None else 0.0,
            "key_opinion":   key_opinion,
        })

        # ── Link comment to its parent post ───────────────────────────────────
        # post_id is always the root post regardless of comment nesting depth.
        # parent_id can be t1_ (reply to comment) or t3_ (reply to post).
        # We always link to root post for graph traversal simplicity.
        post_id = item.get("post_id")
        if post_id:
            session.run(MERGE_COMMENT_ON_POST, {
                "comment_id": item_id,
                "post_id":    post_id,
            })

    # ── Entities: same logic for both posts and comments ─────────────────────
    for entity in entities:
        # Defensive: some items have entities as plain strings ["whey protein"]
        # instead of dicts [{"name": "whey protein", "type": "Supplement"}].
        # Normalise strings to dicts with type "Other" so ingestion never fails.
        if isinstance(entity, str):
            entity = {"name": entity, "type": "Other"}
        name  = entity.get("name", "").strip()
        etype = entity.get("type", "Other").strip()
        if name and etype:
            session.run(MERGE_ENTITY, {
                "item_id":     item_id,
                "name":        name.lower(),   # normalise case for deduplication
                "entity_type": etype,
            })


# ── Batch processor ───────────────────────────────────────────────────────────

def ingest_batch(driver, batch: list[dict]) -> int:
    """
    Ingests one batch of items in a single transaction.
    Returns number of successfully ingested items.

    Using a transaction per batch (not per item) because:
    - One transaction per item = 10k+ round trips to AuraDB = very slow
    - One transaction for everything = if it fails, everything rolls back
    - Batches of 100 = good balance between speed and fault tolerance
    """
    ingested = 0
    with driver.session() as session:
        with session.begin_transaction() as tx:
            for item in batch:
                try:
                    ingest_item(tx, item)
                    ingested += 1
                except Exception as e:
                    log.warning(f"  Failed to ingest item {item.get('id')}: {e}")
                    # Continue with rest of batch — don't abort whole transaction
            tx.commit()
    return ingested


# ── Per-file orchestrator ─────────────────────────────────────────────────────

def ingest_file(driver, filename: str, done_ids: set) -> int:
    """
    Ingests all pending items from one extracted file.
    Returns count of newly ingested items.
    """
    ext_path = EXTRACTED_DIR / f"{filename}.json"

    if not ext_path.exists():
        log.warning(f"  Extracted file not found: {ext_path}")
        return 0

    items = json.load(open(ext_path, encoding="utf-8"))
    log.info(f"\nFILE: {filename} — {len(items)} items")

    # Skip items already ingested (checkpoint)
    pending = [it for it in items if it.get("id") not in done_ids]
    log.info(f"  Pending: {len(pending)} (skipping {len(items)-len(pending)} already done)")

    if not pending:
        log.info("  Nothing to do.")
        return 0

    total_ingested = 0

    # Process in batches
    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE

        log.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} items)...")

        try:
            ingested = ingest_batch(driver, batch)
            total_ingested += ingested

            # Mark batch as done in checkpoint
            for item in batch:
                done_ids.add(item.get("id"))
            save_checkpoint(done_ids)

            log.info(f"  Batch {batch_num} done — {ingested}/{len(batch)} ingested | total: {total_ingested}")

        except Exception as e:
            log.error(f"  Batch {batch_num} failed: {e}")
            log.info("  Skipping batch, continuing...")
            continue

        # Small pause between batches — AuraDB Free has write rate limits
        time.sleep(0.5)

    return total_ingested


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("Hercules Cyclops — Stage 3: Neo4j Graph Ingestion")
    log.info(f"Extracted dir: {EXTRACTED_DIR}")

    # ── Step 1: connect ───────────────────────────────────────────────────────
    driver = get_driver()

    # ── Step 2: schema setup (idempotent — safe to run every time) ───────────
    run_schema_setup(driver)

    # ── Step 3: load checkpoint ───────────────────────────────────────────────
    done_ids = load_checkpoint()
    log.info(f"Checkpoint: {len(done_ids)} items already ingested")

    # ── Step 4: ingest all files ──────────────────────────────────────────────
    grand_total = 0

    for filename in TARGET_FILES:
        try:
            count = ingest_file(driver, filename, done_ids)
            grand_total += count
        except KeyboardInterrupt:
            log.info("\nInterrupted. Progress saved via checkpoint.")
            break
        except Exception as e:
            log.error(f"Fatal error on {filename}: {e}")
            continue

    # ── Step 5: final node counts ─────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("INGESTION COMPLETE")
    log.info(f"Total items ingested this run: {grand_total}")
    log.info("\nNode counts in Neo4j:")

    with driver.session() as session:
        for label in ["Post", "Comment", "Author", "Entity", "Topic", "Subreddit"]:
            count = session.run(
                f"MATCH (n:{label}) RETURN count(n) as c"
            ).single()["c"]
            log.info(f"  {label:<12}: {count:>6} nodes")

        # Relationship counts
        log.info("\nRelationship counts:")
        for rel in ["POSTED", "COMMENTED", "ON", "MENTIONS", "TAGGED", "IN"]:
            count = session.run(
                f"MATCH ()-[r:{rel}]->() RETURN count(r) as c"
            ).single()["c"]
            log.info(f"  {rel:<12}: {count:>6} relationships")

    close_driver(driver)
    log.info("\nStage 3 complete. Proceed to Stage 4 (ChromaDB ingestion).")


if __name__ == "__main__":
    main()
