"""
rrf.py — Hercules Cyclops, Stage 5 (Fusion)
Reciprocal Rank Fusion — merges graph and vector retrieval results.

WHAT THIS FILE DOES:
  Takes two ranked lists (from graph_retriever + vector_retriever) and
  merges them into a single ranked list using Reciprocal Rank Fusion.

WHY RRF (not score averaging):
  Graph scores (relevance × engagement) and vector scores (cosine similarity
  0.0–1.0) live on completely different scales. You cannot average them —
  a graph score of 120.0 and a vector score of 0.65 are not comparable numbers.

  RRF only uses RANK POSITION (1st, 2nd, 3rd...) — it ignores the raw scores
  entirely. This makes it scale-agnostic: it doesn't matter what units each
  retriever uses.

RRF FORMULA:
  For each document d across all result lists:
    RRF_score(d) = Σ 1 / (k + rank(d))

  Where:
    k    = 60 (standard constant from the original RRF paper)
    rank = position in the ranked list (1-indexed)

  Example with k=60:
    Document appears at rank 1 in graph list  → 1/(60+1) = 0.01639
    Document appears at rank 3 in vector list → 1/(60+3) = 0.01587
    Combined RRF score                        → 0.03226

  A document that appears HIGH in BOTH lists gets the biggest boost.
  A document that appears in only one list gets a smaller score.
  This is exactly the behaviour we want — hybrid retrieval should favour
  results that both systems agree are relevant.

WHY k=60:
  The original paper (Cormack et al., 2009) found k=60 optimal across
  many benchmarks. It prevents documents ranked #1 from dominating too
  heavily, giving lower-ranked documents a fair chance to surface
  if they appear consistently across lists.

USAGE:
  from src.fusion.rrf import reciprocal_rank_fusion

  graph_results  = graph_retriever.retrieve(query, top_k=20)
  vector_results = vector_retriever.retrieve(query, top_k=20)
  fused_results  = reciprocal_rank_fusion(graph_results, vector_results, top_k=10)
"""

import logging

log = logging.getLogger(__name__)

# Standard RRF constant from Cormack et al. (2009)
# Increasing k smooths out rank differences; decreasing amplifies rank 1 advantage
K = 60


def reciprocal_rank_fusion(
    *result_lists: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """
    Merges any number of ranked result lists using Reciprocal Rank Fusion.

    Args:
      *result_lists: Any number of ranked lists. Each list is a list of dicts
                     with at least an "id" field. Order = rank (index 0 = rank 1).
      top_k:         Number of results to return after fusion.

    Returns:
      Fused list of result dicts sorted by RRF score descending.
      Each result has an added "rrf_score" field and "sources" list
      showing which retrievers contributed to this result.

    The function is variadic (*result_lists) — works with 2 lists (standard)
    or more if additional retrievers are added in future.
    """
    # rrf_scores[id] = cumulative RRF score across all lists
    rrf_scores: dict[str, float] = {}

    # item_data[id] = the full result dict (from whichever list has best data)
    item_data: dict[str, dict] = {}

    # sources[id] = list of sources that returned this item e.g. ["graph", "vector"]
    sources: dict[str, list] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list, start=1):
            item_id = item.get("id")
            if not item_id:
                continue

            # RRF contribution from this list at this rank
            rrf_contribution = 1.0 / (K + rank)

            # Accumulate score
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + rrf_contribution

            # Keep item data (prefer whichever has longer text — more context)
            if item_id not in item_data:
                item_data[item_id] = item
            else:
                existing_text = item_data[item_id].get("text", "")
                new_text      = item.get("text", "")
                if len(new_text) > len(existing_text):
                    item_data[item_id] = item

            # Track which sources contributed
            source = item.get("source", "unknown")
            if item_id not in sources:
                sources[item_id] = []
            if source not in sources[item_id]:
                sources[item_id].append(source)

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Build final result list
    fused = []
    for item_id in sorted_ids[:top_k]:
        result = dict(item_data[item_id])  # copy to avoid mutating original
        result["rrf_score"] = round(rrf_scores[item_id], 6)
        result["sources"]   = sources[item_id]
        # Mark as hybrid if appeared in multiple sources
        result["source"]    = "hybrid" if len(sources[item_id]) > 1 else sources[item_id][0]
        fused.append(result)

    log.info(
        f"RRF: merged {sum(len(l) for l in result_lists)} results "
        f"({', '.join(str(len(l)) for l in result_lists)}) "
        f"→ {len(fused)} fused results"
    )

    return fused


def format_rrf_results(fused: list[dict], show_text_chars: int = 120) -> str:
    """
    Pretty-prints fused results for logging/debugging.
    Used in demo.py to show results before passing to answer generator.
    """
    lines = []
    for i, r in enumerate(fused, start=1):
        sources_str = "+".join(r.get("sources", ["?"]))
        lines.append(
            f"  [{i}] rrf={r['rrf_score']:.5f} src={sources_str} "
            f"window={r.get('window','')} sub={r.get('subreddit','')} "
            f"sentiment={r.get('sentiment','')}({r.get('sentiment_score',0):.2f})"
        )
        text = r.get("text", "")[:show_text_chars]
        lines.append(f"       {text}...")
    return "\n".join(lines)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Simulate graph results (high scores, different scale)
    graph_results = [
        {"id": "a", "text": "whey protein India review", "score": 120.0,
         "source": "graph", "window": "window_1_recent", "subreddit": "Fitness_India",
         "sentiment": "positive", "sentiment_score": 0.8, "author": "u1",
         "permalink": "http://r/1"},
        {"id": "b", "text": "best supplements for muscle", "score": 90.0,
         "source": "graph", "window": "window_2_midterm", "subreddit": "Fitness_India",
         "sentiment": "positive", "sentiment_score": 0.6, "author": "u2",
         "permalink": "http://r/2"},
        {"id": "c", "text": "protein powder price hike", "score": 60.0,
         "source": "graph", "window": "window_1_recent", "subreddit": "Fitness_India",
         "sentiment": "negative", "sentiment_score": -0.5, "author": "u3",
         "permalink": "http://r/3"},
    ]

    # Simulate vector results (cosine scores 0-1)
    vector_results = [
        {"id": "b", "text": "best supplements for muscle gain in India", "score": 0.85,
         "source": "vector", "window": "window_2_midterm", "subreddit": "Fitness_India",
         "sentiment": "positive", "sentiment_score": 0.6, "author": "u2",
         "permalink": "http://r/2"},
        {"id": "d", "text": "MuscleBlaze vs ON whey comparison", "score": 0.78,
         "source": "vector", "window": "window_1_recent", "subreddit": "Fitness_India",
         "sentiment": "neutral", "sentiment_score": 0.1, "author": "u4",
         "permalink": "http://r/4"},
        {"id": "a", "text": "whey protein India review", "score": 0.72,
         "source": "vector", "window": "window_1_recent", "subreddit": "Fitness_India",
         "sentiment": "positive", "sentiment_score": 0.8, "author": "u1",
         "permalink": "http://r/1"},
    ]

    print("Graph results (raw):")
    for i, r in enumerate(graph_results, 1):
        print(f"  [{i}] id={r['id']} score={r['score']}")

    print("\nVector results (raw):")
    for i, r in enumerate(vector_results, 1):
        print(f"  [{i}] id={r['id']} score={r['score']}")

    fused = reciprocal_rank_fusion(graph_results, vector_results, top_k=5)

    print("\nFused results (RRF):")
    for i, r in enumerate(fused, 1):
        print(f"  [{i}] id={r['id']} rrf={r['rrf_score']:.5f} "
              f"source={r['source']} sources={r['sources']}")

    print("\nExpected: 'b' ranks highest (appears in both lists at high rank)")
    print("Expected: 'a' second (appears in both lists)")
    print("Expected: 'c' and 'd' lower (appear in only one list)")

    print(f"\n{format_rrf_results(fused)}")
