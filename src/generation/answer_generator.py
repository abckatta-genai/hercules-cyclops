"""
answer_generator.py — Hercules Cyclops, Stage 6
Generates cited, temporal-aware answers using Claude Sonnet.

WHAT THIS FILE DOES:
  Takes fused retrieval results (from rrf.py) and a user query,
  formats the results as numbered source context, and calls Claude
  Sonnet to synthesize a coherent answer with citations.

WHY CLAUDE SONNET (not Groq) FOR GENERATION:
  Answer generation is the user-facing output. It requires:
    - Nuanced synthesis of multiple sources with conflicting opinions
    - Proper citation formatting ([1], [2], etc.)
    - Temporal reasoning ("in window_1 vs window_3...")
    - Understanding of what's relevant vs noise in the context
  Claude Sonnet outperforms smaller models on all of these tasks.
  Groq/llama-3.1-8b was used for extraction (schema-filling) where
  reasoning depth doesn't matter. Generation is different.

BUDGET CONSERVATION:
  Claude Sonnet is called ONLY at query time — never during ingestion.
  Each query uses ~1k-2k tokens. At ~$3/M tokens, budget lasts ~500 queries.
  Context is capped at top 10 fused results to keep token usage predictable.

CITATION FORMAT:
  Sources are numbered [1], [2], ... in the context.
  Claude is instructed to cite by number inline in its answer.
  Each answer ends with a Sources section listing window + permalink.

TEMPORAL AWARENESS:
  The prompt explicitly tells Claude about the three time windows and
  instructs it to note when findings are window-specific.
  This is what enables answers like "In the recent window, sentiment
  shifted toward [X], whereas historically [Y] was dominant."

USAGE:
  from src.generation.answer_generator import AnswerGenerator
  generator = AnswerGenerator()
  answer = generator.generate(query, fused_results)
  print(answer)
"""

import logging
import os
from pathlib import Path

import anthropic

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# Claude Sonnet — best balance of quality and cost for generation
# Use sparingly — only at query time, never during ingestion
CLAUDE_MODEL   = "claude-sonnet-4-6"
MAX_TOKENS     = 1024   # enough for a thorough cited answer
MAX_SOURCES    = 10     # cap context to top 10 results to control token usage

# Time window labels for human-readable output
WINDOW_LABELS = {
    "window_1_recent":     "Recent (last 30 days)",
    "window_2_midterm":    "Mid-term (31–90 days ago)",
    "window_3_historical": "Historical (91–180 days ago)",
}


class AnswerGenerator:
    """
    Generates cited answers using Claude Sonnet.

    Usage:
      generator = AnswerGenerator()
      answer = generator.generate(
          query="How has whey protein sentiment changed over time?",
          fused_results=rrf_results,
      )
      print(answer)
    """

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set.\n"
                "Export from .env: export $(grep ANTHROPIC_API_KEY .env | xargs)"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        log.info("AnswerGenerator initialised (Claude Sonnet).")

    def generate(
        self,
        query: str,
        fused_results: list[dict],
        query_type: str = "general",
    ) -> str:
        """
        Generates a cited answer for the given query using fused results as context.

        Args:
          query:        The user's question
          fused_results: Ranked list from rrf.reciprocal_rank_fusion()
          query_type:   Hint for answer framing:
                        "general"     — standard synthesis
                        "temporal"    — emphasise time-window comparisons
                        "entity"      — emphasise named entities / brands
                        "sentiment"   — emphasise sentiment trends

        Returns:
          String containing the answer with inline citations and sources section.
        """
        if not fused_results:
            return "No relevant results found for this query."

        # Cap to top MAX_SOURCES results
        sources = fused_results[:MAX_SOURCES]

        # Build context block
        context = self._format_context(sources)

        # Build prompt
        prompt = self._build_prompt(query, context, query_type)

        log.info(f"Calling Claude Sonnet for query: '{query}'")
        log.info(f"Context: {len(sources)} sources, ~{len(prompt)//4} tokens")

        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.content[0].text.strip()

            # Append sources footer
            answer += "\n\n" + self._format_sources_footer(sources)

            log.info(f"Answer generated ({len(answer)} chars)")
            return answer

        except Exception as e:
            log.error(f"Claude API call failed: {e}")
            return f"Answer generation failed: {e}"

    def _format_context(self, sources: list[dict]) -> str:
        """
        Formats fused results as numbered source context for Claude.

        Each source includes:
          - Number (for citation)
          - Time window (human-readable label)
          - Subreddit
          - Sentiment + score
          - The actual text content
        """
        lines = []
        for i, source in enumerate(sources, start=1):
            window_label = WINDOW_LABELS.get(
                source.get("window", ""),
                source.get("window", "Unknown window")
            )
            sentiment       = source.get("sentiment", "neutral")
            sentiment_score = source.get("sentiment_score", 0.0)
            subreddit       = source.get("subreddit", "")
            text            = (source.get("text", "") or "").strip()

            # Truncate very long texts to keep context manageable
            if len(text) > 500:
                text = text[:500] + "..."

            lines.append(
                f"[{i}] {window_label} | r/{subreddit} | "
                f"Sentiment: {sentiment} ({sentiment_score:+.2f})\n"
                f"    {text}"
            )

        return "\n\n".join(lines)

    def _build_prompt(self, query: str, context: str, query_type: str) -> str:
        """
        Builds the Claude prompt with query, context, and instructions.

        The prompt is structured to:
          1. Give Claude role context (market research analyst)
          2. Explain the three time windows
          3. Provide numbered sources
          4. Ask for cited, temporal-aware answer
          5. Specify output format
        """
        temporal_instruction = ""
        if query_type == "temporal":
            temporal_instruction = """
Pay special attention to HOW opinions or trends CHANGED across time windows.
Compare the Recent window vs the Historical window explicitly.
Note if something emerged, disappeared, or intensified over time."""

        elif query_type == "sentiment":
            temporal_instruction = """
Focus on the sentiment scores and labels across time windows.
Identify if overall sentiment shifted positively, negatively, or stayed neutral.
Note any communities (subreddits) with notably different sentiment."""

        elif query_type == "entity":
            temporal_instruction = """
Focus on named entities (brands, products, gyms, supplements) mentioned.
Note which entities appear most frequently and with what sentiment.
Identify if any entities rose or fell in prominence across time windows."""

        return f"""You are a market research analyst for an AI consumer intelligence platform.
You are analysing Reddit discussions about gym culture and fitness habits in urban India.

TIME WINDOWS (the data spans three periods):
- Recent (window_1_recent): Last 30 days — current conversation
- Mid-term (window_2_midterm): 31–90 days ago — emerging patterns
- Historical (window_3_historical): 91–180 days ago — baseline

USER QUERY:
{query}

RETRIEVED SOURCES (cite by number inline using [1], [2], etc.):
{context}
{temporal_instruction}

INSTRUCTIONS:
- Answer the query using ONLY the information in the sources above
- Cite sources inline using [1], [2], etc. after each claim
- Note which time window findings come from when relevant
- If sources conflict, acknowledge the disagreement
- Be specific — use actual terms, brands, sentiments from the sources
- Keep the answer focused and under 300 words
- Do NOT make up information not present in the sources

Answer:"""

    def _format_sources_footer(self, sources: list[dict]) -> str:
        """Formats a sources list for the end of the answer."""
        lines = ["**Sources:**"]
        for i, source in enumerate(sources, start=1):
            window_label = WINDOW_LABELS.get(
                source.get("window", ""),
                source.get("window", "")
            )
            permalink = source.get("permalink", "")
            subreddit = source.get("subreddit", "")
            lines.append(f"[{i}] {window_label} | r/{subreddit} | {permalink}")
        return "\n".join(lines)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Minimal test with mock results — just verifies Claude API connectivity
    mock_results = [
        {
            "id": "test1",
            "text": "MuscleBlaze whey protein has seen a 40% price hike in the last month. Many users are switching to OWN or Nakpro as budget alternatives.",
            "window": "window_1_recent",
            "subreddit": "Fitness_India",
            "sentiment": "negative",
            "sentiment_score": -0.6,
            "source": "hybrid",
            "sources": ["graph", "vector"],
            "rrf_score": 0.032,
            "permalink": "https://reddit.com/r/Fitness_India/comments/test1",
        },
        {
            "id": "test2",
            "text": "Whey prices in India were stable 3-6 months ago. MuscleBlaze and Optimum Nutrition were the top recommendations.",
            "window": "window_3_historical",
            "subreddit": "Fitness_India",
            "sentiment": "positive",
            "sentiment_score": 0.5,
            "source": "graph",
            "sources": ["graph"],
            "rrf_score": 0.016,
            "permalink": "https://reddit.com/r/Fitness_India/comments/test2",
        },
    ]

    generator = AnswerGenerator()
    answer = generator.generate(
        query="How has whey protein pricing and sentiment changed over the last 6 months?",
        fused_results=mock_results,
        query_type="temporal",
    )
    print("\n" + "="*60)
    print("GENERATED ANSWER:")
    print("="*60)
    print(answer)
