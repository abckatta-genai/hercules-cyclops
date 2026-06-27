"""
graph_retriever.py — Hercules Cyclops, Stage 5
Retrieves relevant posts/comments from Neo4j using keyword + entity matching.

WHAT THIS FILE DOES:
  Takes a natural language query, extracts keywords from it, and runs
  Cypher queries against Neo4j to find relevant content.

WHY GRAPH RETRIEVAL (not just vector):
  Graph retrieval excels at relationship and entity queries:
  - "Which brands are most discussed?" → count MENTIONS relationships
  - "Who are the most active authors?" → count POSTED relationships
  - "What topics appear in window_1 but not window_3?" → compare across windows
  Vector search would miss these structural patterns entirely.

RETRIEVAL STRATEGY:
  Two Cypher queries run per call:

  1. ENTITY MATCH — finds posts/comments that MENTIONS entities whose
     name contains any query keyword. Scores by mention count + post score.

  2. TOPIC MATCH — finds posts tagged with topics containing query keywords.
     Topics are broader labels ("protein powder", "gym costs") that often
     match query intent even when exact entity names don't.

  Results from both queries are merged and deduplicated before returning.

RETURN FORMAT (same structure as vector_retriever.py for RRF compatibility):
  [
    {
      "id":         str,   # item ID
      "type":       str,   # "post" or "comment"
      "text":       str,   # title+selftext or body
      "score":      float, # relevance score (higher = more relevant)
      "subreddit":  str,
      "window":     str,
      "author":     str,
      "permalink":  str,
      "sentiment":  str,
      "sentiment_score": float,
      "source":     "graph"
    },
    ...
  ]

TEMPORAL FILTERING:
  Pass window= parameter to restrict results to a specific time window.
  e.g. retrieve(query, window="window_1_recent")
  Pass windows= list for multi-window comparison queries.

USAGE:
  from src.retrieval.graph_retriever import GraphRetriever
  retriever = GraphRetriever()
  results = retriever.retrieve("best protein powder India", top_k=10)
  retriever.close()
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Allow imports from src/graph/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph"))
from neo4j_client import get_driver, close_driver

log = logging.getLogger(__name__)

# ── Stop words ────────────────────────────────────────────────────────────────
# Common words that add noise to keyword matching — excluded from entity search
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "about", "how", "what", "which",
    "who", "when", "where", "why", "that", "this", "these", "those",
    "my", "your", "his", "her", "its", "our", "their", "i", "me", "we",
    "you", "he", "she", "it", "they", "them", "us", "india", "indian",
    "gym", "fitness", "reddit", "post", "comment", "subreddit",
}


def extract_keywords(query: str) -> list[str]:
    """
    Extracts meaningful keywords from a natural language query.
    Lowercases, removes punctuation, filters stop words.
    Returns list of keywords sorted by length (longer = more specific first).
    """
    # Lowercase and remove punctuation
    clean = re.sub(r"[^\w\s]", " ", query.lower())
    tokens = clean.split()

    # Filter stop words and short tokens
    keywords = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # Sort by length descending — longer keywords are more specific
    keywords = sorted(set(keywords), key=len, reverse=True)

    return keywords[:10]  # cap at 10 keywords to keep Cypher query manageable


class GraphRetriever:
    """
    Retrieves relevant content from Neo4j using entity and topic matching.

    Usage:
      retriever = GraphRetriever()
      results = retriever.retrieve("whey protein recommendations", top_k=10)
      retriever.close()
    """

    def __init__(self):
        self.driver = get_driver()
        log.info("GraphRetriever initialised.")

    def close(self):
        close_driver(self.driver)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        window: Optional[str] = None,
        windows: Optional[list] = None,
    ) -> list[dict]:
        """
        Main retrieval method.

        Args:
          query:   Natural language query string
          top_k:   Number of results to return
          window:  Filter to single time window e.g. "window_1_recent"
          windows: Filter to list of windows e.g. ["window_1_recent", "window_3_historical"]

        Returns:
          List of result dicts sorted by relevance score descending.
        """
        keywords = extract_keywords(query)
        log.info(f"GraphRetriever: query='{query}' keywords={keywords}")

        if not keywords:
            log.warning("No keywords extracted from query. Returning empty.")
            return []

        # Build window filter for Cypher
        window_filter = self._build_window_filter(window, windows)

        # Run both retrieval strategies
        entity_results = self._retrieve_by_entity(keywords, window_filter, top_k * 2)
        topic_results  = self._retrieve_by_topic(keywords, window_filter, top_k * 2)

        # Merge + deduplicate by ID
        merged = self._merge_results(entity_results, topic_results)

        # Sort by score descending, return top_k
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

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

    def _build_window_filter(
        self,
        window: Optional[str],
        windows: Optional[list],
    ) -> str:
        """Returns a Cypher WHERE clause fragment for window filtering."""
        if window:
            return f"AND n.window = '{window}'"
        elif windows:
            window_list = ", ".join(f"'{w}'" for w in windows)
            return f"AND n.window IN [{window_list}]"
        return ""  # no filter — search all windows

    def _retrieve_by_entity(
        self,
        keywords: list[str],
        window_filter: str,
        limit: int,
    ) -> list[dict]:
        """
        Finds posts/comments that MENTIONS entities matching query keywords.

        Score = number of matching entity mentions × post engagement score.
        Posts with more keyword-matching entities rank higher.
        """
        # Build regex pattern for keyword matching against entity names
        # Uses case-insensitive contains match
        keyword_pattern = "|".join(re.escape(k) for k in keywords)

        cypher = f"""
        MATCH (n)-[:MENTIONS]->(e:Entity)
        WHERE toLower(e.name) =~ '.*({keyword_pattern}).*'
        {window_filter}
        WITH n, count(e) AS mention_count
        ORDER BY mention_count DESC, n.score DESC
        LIMIT {limit}
        RETURN
          n.id            AS id,
          n.window        AS window,
          n.subreddit     AS subreddit,
          n.author        AS author,
          n.permalink     AS permalink,
          n.sentiment     AS sentiment,
          n.sentiment_score AS sentiment_score,
          n.score         AS engagement,
          mention_count,
          CASE WHEN n.title IS NOT NULL THEN 'post' ELSE 'comment' END AS type,
          CASE WHEN n.title IS NOT NULL
               THEN coalesce(n.title, '') + ' ' + coalesce(n.selftext, '')
               ELSE coalesce(n.body, '')
          END AS text
        """

        return self._run_query(cypher, "entity", keywords)

    def _retrieve_by_topic(
        self,
        keywords: list[str],
        window_filter: str,
        limit: int,
    ) -> list[dict]:
        """
        Finds posts TAGGED with topics matching query keywords.
        Topics are broader labels — catches cases where entity names
        don't exactly match but the topic area does.
        """
        keyword_pattern = "|".join(re.escape(k) for k in keywords)

        cypher = f"""
        MATCH (n:Post)-[:TAGGED]->(t:Topic)
        WHERE toLower(t.name) =~ '.*({keyword_pattern}).*'
        {window_filter}
        WITH n, count(t) AS topic_count
        ORDER BY topic_count DESC, n.score DESC
        LIMIT {limit}
        RETURN
          n.id            AS id,
          n.window        AS window,
          n.subreddit     AS subreddit,
          n.author        AS author,
          n.permalink     AS permalink,
          n.sentiment     AS sentiment,
          n.sentiment_score AS sentiment_score,
          n.score         AS engagement,
          topic_count     AS mention_count,
          'post'          AS type,
          coalesce(n.title, '') + ' ' + coalesce(n.selftext, '') AS text
        """

        return self._run_query(cypher, "topic", keywords)

    def _run_query(self, cypher: str, strategy: str, keywords: list) -> list[dict]:
        """Runs a Cypher query and normalises results to standard format."""
        results = []
        try:
            with self.driver.session() as session:
                records = session.run(cypher).data()
                for r in records:
                    # Normalise score: combine mention count with engagement
                    mention_count = r.get("mention_count", 1) or 1
                    engagement    = r.get("engagement", 0) or 0
                    # Score = mentions weighted higher than raw engagement
                    score = float(mention_count * 10 + min(engagement, 100))

                    results.append({
                        "id":              r.get("id", ""),
                        "type":            r.get("type", "post"),
                        "text":            (r.get("text", "") or "").strip(),
                        "score":           score,
                        "subreddit":       r.get("subreddit", ""),
                        "window":          r.get("window", ""),
                        "author":          r.get("author", ""),
                        "permalink":       r.get("permalink", ""),
                        "sentiment":       r.get("sentiment", "neutral"),
                        "sentiment_score": float(r.get("sentiment_score", 0.0) or 0.0),
                        "source":          "graph",
                        "strategy":        strategy,
                    })

            log.info(f"  Graph [{strategy}]: {len(results)} results for keywords={keywords}")

        except Exception as e:
            log.error(f"  Graph [{strategy}] query failed: {e}")

        return results

    def _merge_results(self, *result_lists) -> list[dict]:
        """
        Merges multiple result lists, deduplicating by ID.
        When same ID appears in multiple lists, keeps the higher score.
        """
        merged = {}
        for results in result_lists:
            for r in results:
                item_id = r["id"]
                if item_id not in merged or r["score"] > merged[item_id]["score"]:
                    merged[item_id] = r
        return list(merged.values())


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    retriever = GraphRetriever()

    print("\nTest 1: Entity search — 'whey protein'")
    results = retriever.retrieve("whey protein supplement", top_k=5)
    for i, r in enumerate(results):
        print(f"  [{i+1}] score={r['score']:.1f} window={r['window']} "
              f"subreddit={r['subreddit']} text={r['text'][:80]}...")

    print("\nTest 2: Temporal filter — 'gym membership' in window_1_recent")
    results = retriever.retrieve("gym membership cost", top_k=5, window="window_1_recent")
    for i, r in enumerate(results):
        print(f"  [{i+1}] score={r['score']:.1f} window={r['window']} text={r['text'][:80]}...")

    retriever.close()
