"""
demo.py — Hercules Cyclops, End-to-End Demonstration
Runs 4 queries showing the full hybrid GraphRAG pipeline.

WHAT THIS SCRIPT DOES:
  For each of 4 queries, it:
    1. Runs graph retrieval (Neo4j entity + topic matching)
    2. Runs vector retrieval (ChromaDB semantic search)
    3. Fuses both result lists with Reciprocal Rank Fusion
    4. Generates a cited answer using Claude Sonnet

  For each query, it prints:
    - Graph-only results (what graph retrieval found alone)
    - Vector-only results (what vector retrieval found alone)
    - Fused results (what RRF selected from both)
    - Final Claude-generated answer with citations

THE 4 QUERY TYPES (as required by assignment):
  1. SEMANTIC    — "What do people say about losing motivation at the gym?"
                   Vector-dominant: meaning-based, no exact keywords
  2. GRAPH       — "Which supplement brands are most discussed and by whom?"
                   Graph-dominant: relationship traversal, entity counting
  3. HYBRID      — "What are the most recommended protein supplements in India?"
                   Needs both: semantic matching + entity relationships
  4. TIME-SERIES — "How has sentiment around whey protein changed over 6 months?"
                   Temporal: compares window_1_recent vs window_3_historical

USAGE:
  export $(grep -E 'NEO4J|ANTHROPIC' .env | xargs)
  python demo.py

  Output is printed to stdout. Pipe to file if needed:
  python demo.py > demo_output.txt
"""

import logging
import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow imports from all src/ subdirectories
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "graph"))
sys.path.insert(0, str(ROOT / "src" / "vector"))
sys.path.insert(0, str(ROOT / "src" / "retrieval"))
sys.path.insert(0, str(ROOT / "src" / "fusion"))
sys.path.insert(0, str(ROOT / "src" / "generation"))

from graph_retriever  import GraphRetriever
from vector_retriever import VectorRetriever
from rrf              import reciprocal_rank_fusion, format_rrf_results
from answer_generator import AnswerGenerator

logging.basicConfig(
    level=logging.WARNING,  # suppress info logs during demo for clean output
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── Demo queries ──────────────────────────────────────────────────────────────
QUERIES = [
    {
        "id":          1,
        "type":        "SEMANTIC",
        "label":       "Purely Semantic Query (Vector-Dominant)",
        "query":       "What do people say about losing motivation to work out or feeling burned out at the gym?",
        "query_type":  "sentiment",
        "window":      None,
        "description": "This query has no exact keyword matches for entities or topics. "
                       "Vector retrieval captures semantic meaning (burnout, demotivation) "
                       "even when users wrote 'lost the drive' or 'can't be bothered'.",
    },
    {
        "id":          2,
        "type":        "GRAPH",
        "label":       "Relationship/Traversal Query (Graph-Dominant)",
        "query":       "Which supplement brands are most frequently mentioned and what sentiment do they have?",
        "query_type":  "entity",
        "window":      None,
        "description": "This query needs graph traversal: find Entity nodes of type Brand, "
                       "count MENTIONS relationships, aggregate sentiment across posts. "
                       "Pure vector search cannot count relationships.",
    },
    {
        "id":          3,
        "type":        "HYBRID",
        "label":       "Hybrid Query (Needs Both Graph + Vector)",
        "query":       "What protein supplements do Indian gym-goers recommend for muscle building on a budget?",
        "query_type":  "entity",
        "window":      None,
        "description": "Needs graph retrieval to find posts mentioning supplement entities, "
                       "AND vector retrieval to find semantically similar content about "
                       "budget recommendations even when exact terms differ.",
    },
    {
        "id":          4,
        "type":        "TIME-SERIES",
        "label":       "Time-Series Comparison Query",
        "query":       "How has sentiment and discussion around whey protein prices changed over the last 6 months?",
        "query_type":  "temporal",
        "window":      None,
        "description": "Compares window_1_recent vs window_3_historical to show how "
                       "whey protein pricing sentiment evolved. Requires temporal metadata "
                       "on both graph nodes and vector metadata filters.",
    },
]

TOP_K_RETRIEVAL = 15   # retrieve more than needed, RRF will select best
TOP_K_FUSED     = 10   # final fused results passed to answer generator
SEPARATOR       = "=" * 70


def print_results(results: list[dict], label: str, max_show: int = 5) -> None:
    """Pretty-prints retrieval results."""
    print(f"\n  {label} ({len(results)} results, showing top {min(max_show, len(results))}):")
    for i, r in enumerate(results[:max_show], 1):
        score = r.get("rrf_score", r.get("score", 0))
        score_str = f"{score:.5f}" if r.get("rrf_score") else f"{score:.3f}"
        print(f"    [{i}] score={score_str} | window={r.get('window','')} | "
              f"sub=r/{r.get('subreddit','')} | sentiment={r.get('sentiment','')}({r.get('sentiment_score',0):+.2f})")
        text = (r.get("text", "") or "")[:100]
        print(f"         {text}...")


def run_demo():
    print(SEPARATOR)
    print("HERCULES CYCLOPS — Hybrid GraphRAG Demo")
    print("Topic: Gym culture and fitness habits in urban India")
    print("Data:  Reddit (5 subreddits × 3 time windows = ~11k posts+comments)")
    print(SEPARATOR)

    # ── Initialise all components once ───────────────────────────────────────
    print("\nInitialising pipeline components...")
    graph_retriever  = GraphRetriever()
    vector_retriever = VectorRetriever()
    generator        = AnswerGenerator()
    print("All components ready.\n")

    # ── Run each query ────────────────────────────────────────────────────────
    for q in QUERIES:
        print(f"\n{SEPARATOR}")
        print(f"QUERY {q['id']}/4 — {q['type']}")
        print(f"{q['label']}")
        print(SEPARATOR)
        print(f"\nQuestion: {q['query']}")
        print(f"Why this type: {q['description']}")

        # ── Step 1: Graph retrieval ───────────────────────────────────────────
        print(f"\n{'─'*50}")
        print("STEP 1: Graph Retrieval (Neo4j)")
        graph_results = graph_retriever.retrieve(
            query=q["query"],
            top_k=TOP_K_RETRIEVAL,
            window=q.get("window"),
        )
        print_results(graph_results, "Graph-only results")

        # ── Step 2: Vector retrieval ──────────────────────────────────────────
        print(f"\n{'─'*50}")
        print("STEP 2: Vector Retrieval (ChromaDB)")
        vector_results = vector_retriever.retrieve(
            query=q["query"],
            top_k=TOP_K_RETRIEVAL,
            window=q.get("window"),
        )
        print_results(vector_results, "Vector-only results")

        # ── Step 3: RRF fusion ────────────────────────────────────────────────
        print(f"\n{'─'*50}")
        print("STEP 3: Reciprocal Rank Fusion")
        fused_results = reciprocal_rank_fusion(
            graph_results,
            vector_results,
            top_k=TOP_K_FUSED,
        )
        hybrid_count = sum(1 for r in fused_results if r.get("source") == "hybrid")
        print(f"\n  Fused {len(graph_results)} graph + {len(vector_results)} vector "
              f"→ {len(fused_results)} results ({hybrid_count} hybrid)")
        print_results(fused_results, "Fused results (RRF)")

        # ── Step 4: Answer generation ─────────────────────────────────────────
        print(f"\n{'─'*50}")
        print("STEP 4: Claude Sonnet Answer Generation")
        print("\n  Generating answer...\n")

        answer = generator.generate(
            query=q["query"],
            fused_results=fused_results,
            query_type=q["query_type"],
        )

        print("  ANSWER:")
        print("  " + "\n  ".join(answer.split("\n")))

    # ── Wrap up ───────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("DEMO COMPLETE")
    print(f"{SEPARATOR}")
    print("\nPipeline summary:")
    print("  Stage 1: Reddit scraper (Arctic Shift API) → 11,536 raw items")
    print("  Stage 2: LLM extraction (Groq llama-3.1-8b) → entities, sentiment, topics")
    print("  Stage 3: Neo4j graph → 11k nodes, 54k relationships")
    print("  Stage 4: ChromaDB vectors → 11k embeddings (all-MiniLM-L6-v2)")
    print("  Stage 5: Graph + Vector retrieval → parallel ranked lists")
    print("  Stage 6: RRF fusion + Claude Sonnet → cited temporal answers")

    # ── Clean up ──────────────────────────────────────────────────────────────
    graph_retriever.close()


if __name__ == "__main__":
    # Verify required env vars before starting
    missing = []
    for var in ["NEO4J_URI", "NEO4J_PASSWORD", "ANTHROPIC_API_KEY"]:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        print(f"Missing environment variables: {missing}")
        print("Export from .env: export $(grep -E 'NEO4J|ANTHROPIC' .env | xargs)")
        sys.exit(1)

    run_demo()
