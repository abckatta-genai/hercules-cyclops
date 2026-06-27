"""
neo4j_client.py — Hercules Cyclops, Stage 3
Neo4j AuraDB connection and schema initialisation.

WHAT THIS FILE DOES:
  - Connects to Neo4j AuraDB Free using credentials from .env
  - Creates uniqueness constraints on all node types
  - Creates indexes on frequently queried properties
  - Provides a single get_driver() function used by graph_ingestor.py

WHY CONSTRAINTS BEFORE INGESTION:
  MERGE statements (used in graph_ingestor.py) need uniqueness constraints
  to work correctly. Without a constraint on Post.id, two concurrent MERGE
  calls for the same post could both "miss" the existing node and create
  duplicates. Constraints enforce uniqueness at the database level.

WHY INDEXES:
  Queries in graph_retriever.py filter by window, subreddit, and sentiment.
  Without indexes, Neo4j scans every node of that type (full scan).
  With indexes, it jumps directly to matching nodes (O(log n)).

USAGE:
  from src.graph.neo4j_client import get_driver, close_driver, run_schema_setup
  driver = get_driver()
  run_schema_setup(driver)
  # ... use driver ...
  close_driver(driver)
"""

import os
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

log = logging.getLogger(__name__)

# ── Connection ────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j+s://27b2b980.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME",  "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD",  "")


def get_driver():
    """
    Returns a Neo4j driver instance connected to AuraDB.

    neo4j+s:// = encrypted connection (required by AuraDB Free).
    The driver manages a connection pool internally — create once,
    reuse across all ingestion calls, close at the end.
    """
    if not NEO4J_PASSWORD:
        raise RuntimeError(
            "NEO4J_PASSWORD not set in environment.\n"
            "Export from .env: export $(grep NEO4J .env | xargs)"
        )

    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )
        driver.verify_connectivity()
        log.info(f"Connected to Neo4j: {NEO4J_URI}")
        return driver

    except AuthError:
        raise RuntimeError(
            "Neo4j authentication failed.\n"
            "Check NEO4J_USERNAME and NEO4J_PASSWORD in .env"
        )
    except ServiceUnavailable:
        raise RuntimeError(
            f"Neo4j unreachable at {NEO4J_URI}.\n"
            "Check your internet connection and AuraDB instance status."
        )


def close_driver(driver) -> None:
    """Closes the driver and releases all connections."""
    if driver:
        driver.close()
        log.info("Neo4j driver closed.")


# ── Schema: constraints + indexes ────────────────────────────────────────────
# Safe to re-run — IF NOT EXISTS is idempotent.

CONSTRAINTS = [
    # Uniqueness constraints — required for MERGE to work correctly.
    # Each constraint also implicitly creates an index on that property.
    "CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT comment_id IF NOT EXISTS FOR (c:Comment) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT subreddit_name IF NOT EXISTS FOR (s:Subreddit) REQUIRE s.name IS UNIQUE",

    # Entity uniqueness: same name + same type = same node across all posts.
    # "whey protein" as Supplement = one node, not one per post that mentions it.
    "CREATE CONSTRAINT entity_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",

    # Topic uniqueness: same topic string = same node across all windows.
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
]

INDEXES = [
    # window index — most temporal queries filter by window first
    "CREATE INDEX post_window IF NOT EXISTS FOR (p:Post) ON (p.window)",
    "CREATE INDEX comment_window IF NOT EXISTS FOR (c:Comment) ON (c.window)",

    # created_utc — for date range queries within a window
    "CREATE INDEX post_created IF NOT EXISTS FOR (p:Post) ON (p.created_utc)",
    "CREATE INDEX comment_created IF NOT EXISTS FOR (c:Comment) ON (c.created_utc)",

    # sentiment — for filtering by sentiment in graph_retriever.py
    "CREATE INDEX post_sentiment IF NOT EXISTS FOR (p:Post) ON (p.sentiment)",

    # subreddit on Post — for community-specific queries
    "CREATE INDEX post_subreddit IF NOT EXISTS FOR (p:Post) ON (p.subreddit)",

    # entity type — for queries like "find all Brand entities"
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
]


def run_schema_setup(driver) -> None:
    """
    Creates all constraints and indexes.
    Safe to run multiple times — IF NOT EXISTS prevents errors on re-run.
    Always run this ONCE before ingestion starts.
    """
    log.info("Setting up Neo4j schema...")

    with driver.session() as session:
        for cypher in CONSTRAINTS:
            session.run(cypher)
            log.info(f"  Constraint OK: {cypher.split('FOR')[1].split('REQUIRE')[0].strip()}")

        for cypher in INDEXES:
            session.run(cypher)
            log.info(f"  Index OK: {cypher.split('FOR')[1].split('ON')[0].strip()}")

    log.info("Schema setup complete.")


# ── Connectivity test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run directly to test connectivity before running graph_ingestor.py:
      python src/graph/neo4j_client.py

    Expected output:
      Connected to Neo4j: neo4j+s://27b2b980.databases.neo4j.io
      Schema setup complete.
      Post: 0  Comment: 0  Author: 0  ...  (all zero before ingestion)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    driver = get_driver()
    run_schema_setup(driver)

    with driver.session() as session:
        for label in ["Post", "Comment", "Author", "Entity", "Topic", "Subreddit"]:
            count = session.run(
                f"MATCH (n:{label}) RETURN count(n) as c"
            ).single()["c"]
            print(f"  {label}: {count} nodes")

    close_driver(driver)
