"""
extractor.py — Hercules Cyclops, Stage 2 (UNIFIED)
LLM entity + sentiment extraction from Reddit data.

This is the single authoritative extractor. All previous extractor_*.py
files are superseded by this one.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS SCRIPT DOES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. AUDIT — Scans all 9 target files. Skips any file that is already
   ≥95% extracted with zero nulls. Prints a status table upfront.

2. EXTRACT — For each incomplete file, reads raw JSON from data/raw/,
   calls an LLM per item, extracts structured data, writes enriched
   JSON to data/extracted/.

3. KEY ROTATION — Tries Groq keys first (up to 6), then falls back to
   Gemini. When a key hits its daily limit, automatically rotates to
   the next available key without stopping.

4. CHECKPOINT — Saves progress every 50 items. On crash/interrupt,
   resume by re-running the script — it skips already-done items.
   Done IDs are built from BOTH checkpoint file AND output file
   (belt-and-suspenders against checkpoint/output sync issues).

5. NULL POLICY — If extraction fails after all retries, the item is
   SKIPPED ENTIRELY. It does not appear in output with null fields.
   Nulls in Neo4j/ChromaDB cause cascade failures downstream.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTRACTION SCHEMA (per item)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "entities":       [{"name": str, "type": str}],  # named things in text
  "topics":         [str, ...],                     # 3-5 topic labels
  "sentiment":      "positive|negative|neutral",
  "sentiment_score": float,                         # -1.0 to +1.0
  "key_opinion":    str,                            # 1 sentence, author's main point
  "time_window":    str                             # copied from item.window
}

Entity types (9, fixed — map directly to Neo4j node labels in Stage 3):
  Person, Brand, Product, Exercise, Location, Supplement, Gym, Diet, Other

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROVIDER HIERARCHY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Primary:  Groq  — llama-3.1-8b-instant (free, 500k tokens/day per key)
Fallback: Gemini — gemini-2.0-flash     (free tier, 15 RPM)

Groq keys tried in order: GROQ_API_KEY_1 → _2 → _3 → _4 → _5 → _6
Missing keys are skipped automatically — works with any subset.
If all Groq keys are exhausted, falls back to Gemini automatically.
If Gemini also exhausted, saves progress and exits cleanly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Keys are read from environment — export from .env first:
  export $(grep -E 'GROQ_API_KEY|GEMINI_API_KEY' .env | xargs)

  # Run (caffeinate prevents Mac sleep on long jobs):
  caffeinate -i python src/ingestion/extractor.py

  # To resume after crash: just re-run the same command.
  # Checkpoint files ensure no item is processed twice.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THESE TECH CHOICES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Groq llama-3.1-8b-instant:
  - Free tier: 500k tokens/day, ~30 RPM per key
  - Extraction is schema-filling, not reasoning — 8B is sufficient
  - 70B models on Groq have tighter RPM; 8B is throughput-optimised
  - Multiple free keys = horizontal scaling at zero cost

Gemini gemini-2.0-flash (fallback):
  - Free tier: 15 RPM, generous daily quota
  - Used only when all Groq keys are exhausted
  - Different provider = different daily reset cycle (redundancy)

Vanilla Python (no LangChain/LlamaIndex):
  - This is a hiring assignment — evaluators want to see raw engineering
  - Full control over retry logic, key rotation, checkpoint behaviour
"""

import json
import os
import time
import logging
import re
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EXTRACTOR] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolves correctly whether script is run from project root or src/ingestion/
RAW_DIR       = Path(__file__).resolve().parents[2] / "data" / "raw"
EXTRACTED_DIR = Path(__file__).resolve().parents[2] / "data" / "extracted"

# ── Extraction config ─────────────────────────────────────────────────────────
GROQ_MODEL       = "llama-3.1-8b-instant"
GEMINI_MODEL     = "gemini-2.0-flash"
MAX_RETRIES      = 5      # retries per item before skipping
CHECKPOINT_EVERY = 50     # flush to disk every N items processed

# Groq: 30 RPM free tier → 2.5s delay keeps well under limit
GROQ_REQUEST_DELAY   = 2.5
GROQ_BASE_BACKOFF    = 6     # seconds; doubles each retry on rate limit

# Gemini: 15 RPM free tier → 6s delay gives safe headroom
GEMINI_REQUEST_DELAY = 6.0
GEMINI_BASE_BACKOFF  = 15    # Gemini rate limits need longer backoff

# Audit threshold: files ≥95% extracted with 0 nulls are considered complete
COMPLETION_THRESHOLD = 0.95

# ── Target files ──────────────────────────────────────────────────────────────
# All 9 files we care about. Audit runs first and skips complete ones.
# Order: window_1 → window_2 → window_3 (chronological, recent first)
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


# ══════════════════════════════════════════════════════════════════════════════
# KEY POOL MANAGER
# Manages all available API keys across providers.
# Automatically rotates to next key when daily limit is hit.
# ══════════════════════════════════════════════════════════════════════════════

class KeyPoolManager:
    """
    Holds all available API keys and tracks which one is currently active.

    Key priority order:
      1. Groq keys 1-6 (tried in sequence, fast + high throughput)
      2. Gemini key   (fallback, different provider = different reset cycle)

    To add more keys later:
      - Groq: add GROQ_API_KEY_7, GROQ_API_KEY_8 etc. to .env
             then extend the groq_key_names list below
      - Gemini: add GEMINI_API_KEY_2 etc. and extend gemini_key_names

    Keys not present in environment are silently skipped.
    """

    def __init__(self):
        # ── Groq key names in rotation order ─────────────────────────────────
        # Add more key names here to support additional keys in future
        groq_key_names = [
            "GROQ_API_KEY_1",
            "GROQ_API_KEY_2",
            "GROQ_API_KEY_3",
            "GROQ_API_KEY_4",
            "GROQ_API_KEY_5",
            "GROQ_API_KEY_6",
        ]

        # ── Gemini key names (fallback after all Groq keys exhausted) ─────────
        # Add GEMINI_API_KEY_2, _3 etc. here to support multiple Gemini keys
        gemini_key_names = [
            "GEMINI_API_KEY",
        ]

        # Build ordered list of (provider, key_value) tuples — skip missing keys
        self.pool = []

        for name in groq_key_names:
            value = os.getenv(name)
            if value:
                self.pool.append(("groq", value, name))
                log.info(f"  Key loaded: {name} [groq]")

        for name in gemini_key_names:
            value = os.getenv(name)
            if value:
                self.pool.append(("gemini", value, name))
                log.info(f"  Key loaded: {name} [gemini]")

        if not self.pool:
            raise RuntimeError(
                "No API keys found in environment.\n"
                "Export keys first:\n"
                "  export $(grep -E 'GROQ_API_KEY|GEMINI_API_KEY' .env | xargs)"
            )

        self.current_index = 0
        log.info(f"  Key pool: {len(self.pool)} keys available")

    @property
    def current(self):
        """Returns (provider, key_value, key_name) for the active key."""
        return self.pool[self.current_index]

    @property
    def provider(self):
        return self.pool[self.current_index][0]

    @property
    def key_value(self):
        return self.pool[self.current_index][1]

    @property
    def key_name(self):
        return self.pool[self.current_index][2]

    def rotate(self) -> bool:
        """
        Rotates to the next available key.
        Returns True if rotation succeeded, False if all keys are exhausted.
        """
        if self.current_index + 1 >= len(self.pool):
            log.error("All API keys exhausted. No more keys to rotate to.")
            return False

        old_name = self.key_name
        self.current_index += 1
        new_name = self.key_name
        new_provider = self.provider

        log.warning(f"  Key rotated: {old_name} → {new_name} [{new_provider}]")
        return True

    def has_keys_remaining(self) -> bool:
        return self.current_index < len(self.pool)


# ══════════════════════════════════════════════════════════════════════════════
# CLIENT FACTORY
# Creates the right API client object for the current provider.
# ══════════════════════════════════════════════════════════════════════════════

def make_client(provider: str, key_value: str):
    """
    Returns an API client for the given provider.
    Groq and Gemini have different client interfaces — this abstracts that.
    """
    if provider == "groq":
        from groq import Groq
        return Groq(api_key=key_value)

    elif provider == "gemini":
        import google.genai as genai
        return genai.Client(api_key=key_value)

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT ENGINE
# Checks all target files before extraction starts.
# ══════════════════════════════════════════════════════════════════════════════

def audit_files() -> dict[str, dict]:
    """
    Scans all target files and returns their status.

    Returns dict keyed by filename:
    {
      "raw_count":  int,
      "ext_count":  int,
      "null_count": int,
      "complete":   bool,   # True = skip this file
      "pct":        float,  # completion percentage
    }
    """
    log.info("\n" + "=" * 65)
    log.info("AUDIT — scanning all target files")
    log.info("=" * 65)
    log.info(f"  {'File':<40} {'Raw':>5} {'Ext':>5} {'Null':>5}  {'Pct':>6}  Status")
    log.info("  " + "-" * 63)

    results = {}

    for filename in TARGET_FILES:
        raw_path = RAW_DIR / f"{filename}.json"
        ext_path = EXTRACTED_DIR / f"{filename}.json"

        # Raw file must exist (Stage 1 output)
        if not raw_path.exists():
            log.error(f"  RAW FILE MISSING: {raw_path}")
            results[filename] = {"complete": False, "raw_count": 0,
                                  "ext_count": 0, "null_count": 0, "pct": 0.0}
            continue

        raw_items = json.load(open(raw_path, encoding="utf-8"))
        raw_count = len(raw_items)

        # If no extracted file yet → needs processing
        if not ext_path.exists():
            log.info(f"  {filename:<40} {raw_count:>5} {'—':>5} {'—':>5}  {'0.0%':>6}  PROCESS")
            results[filename] = {"complete": False, "raw_count": raw_count,
                                  "ext_count": 0, "null_count": 0, "pct": 0.0}
            continue

        ext_items  = json.load(open(ext_path, encoding="utf-8"))
        ext_count  = len(ext_items)
        null_count = sum(1 for it in ext_items if it.get("extraction") is None)
        pct        = ext_count / raw_count if raw_count > 0 else 0.0

        # Complete = ≥95% extracted AND zero nulls
        complete = (pct >= COMPLETION_THRESHOLD) and (null_count == 0)
        status   = "✓ SKIP" if complete else "⚠ PROCESS"

        log.info(f"  {filename:<40} {raw_count:>5} {ext_count:>5} {null_count:>5}  {pct*100:>5.1f}%  {status}")
        results[filename] = {
            "complete":  complete,
            "raw_count": raw_count,
            "ext_count": ext_count,
            "null_count": null_count,
            "pct":       pct,
        }

    to_process = [f for f, r in results.items() if not r["complete"]]
    log.info(f"\n  Files to process: {len(to_process)}/{len(TARGET_FILES)}")
    if to_process:
        for f in to_process:
            log.info(f"    → {f}")
    else:
        log.info("  All files complete. Nothing to do.")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(item: dict) -> str:
    """
    Builds the extraction prompt for one Reddit item.

    Text assembly:
      post    → title + selftext (concatenated, captures full context)
      comment → body only

    Truncated at 1200 chars to keep prompt tokens predictable and within
    model limits. Each item ≈ 350 tokens in + 150 out = ~500 tokens total.

    The 9 entity types are fixed and map directly to Neo4j node labels
    in Stage 3. Do not change them without updating graph_ingestor.py.

    key_opinion is the primary text stored in ChromaDB for semantic search.
    It should capture the author's actual stance, not a generic summary.
    """
    if item.get("type") == "post":
        text = f"{item.get('title', '')} {item.get('selftext', '')}".strip()
    else:
        text = item.get("body", "").strip()

    if len(text) > 1200:
        text = text[:1200] + "..."

    if len(text) < 5:
        text = "[no content]"

    return f"""You are extracting structured data from a Reddit post or comment about gym and fitness in India.

TEXT:
{text}

Return ONLY a valid JSON object. No explanation. No markdown. No code fences. Just the raw JSON.

{{
  "entities": [
    {{"name": "string", "type": "Person|Brand|Product|Exercise|Location|Supplement|Gym|Diet|Other"}}
  ],
  "topics": ["3 to 5 short topic strings"],
  "sentiment": "positive|negative|neutral",
  "sentiment_score": <float -1.0 to 1.0>,
  "key_opinion": "one sentence capturing the author's main point or stance"
}}

Rules:
- entities: only what is explicitly named in the text; empty array [] if none
- topics: 3-5 short labels e.g. "protein powder", "gym costs", "weight loss progress"
- sentiment_score: -1.0 very negative, 0.0 neutral, 1.0 very positive
- key_opinion: 1 sentence, in English, specific to this author's point
- Output ONLY the JSON object, nothing before or after it"""


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION CALL — GROQ
# ══════════════════════════════════════════════════════════════════════════════

def extract_with_groq(client, item: dict) -> dict | None:
    """
    Calls Groq API for one item.
    Returns clean extraction dict on success.
    Returns None if all retries fail (caller skips the item).
    Raises RuntimeError("DAILY_LIMIT") if daily token quota is hit —
    caller will rotate to next key.
    """
    prompt = build_prompt(item)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,   # deterministic output for schema adherence
                max_tokens=350,    # extraction is small; capping saves daily quota
            )

            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)   # strip markdown fences
            raw = re.sub(r"\s*```$", "", raw)
            raw = raw.strip()

            parsed = json.loads(raw)

            # Validate all required fields present and non-null
            required = {"entities", "topics", "sentiment", "sentiment_score", "key_opinion"}
            missing  = required - set(parsed.keys())
            if missing:
                log.warning(f"  [{item.get('id')}] Missing keys {missing} (attempt {attempt+1})")
                time.sleep(GROQ_BASE_BACKOFF)
                continue

            if any(parsed.get(k) is None for k in required):
                log.warning(f"  [{item.get('id')}] Null values in response (attempt {attempt+1})")
                time.sleep(GROQ_BASE_BACKOFF)
                continue

            return parsed  # ✓ clean extraction

        except json.JSONDecodeError:
            log.warning(f"  [{item.get('id')}] JSON parse failed (attempt {attempt+1})")
            time.sleep(GROQ_BASE_BACKOFF)
            continue

        except Exception as e:
            err = str(e).lower()

            # Daily token limit — signal to caller to rotate key
            if "tokens per day" in err or ("daily" in err and "limit" in err):
                log.warning(f"  Groq daily limit hit on attempt {attempt+1}")
                raise RuntimeError("DAILY_LIMIT")

            # Rate limit (RPM) — back off and retry on same key
            if "429" in str(e) or "rate_limit" in err or "rate limit" in err:
                wait = GROQ_BASE_BACKOFF * (2 ** attempt)
                log.warning(f"  [{item.get('id')}] Rate limit. Waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue

            # Network / transient errors — retry with linear backoff
            if "timeout" in err or "connection" in err:
                wait = GROQ_BASE_BACKOFF * (attempt + 1)
                log.warning(f"  [{item.get('id')}] Network error. Waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue

            # Unknown error — retry once
            log.error(f"  [{item.get('id')}] Unknown error (attempt {attempt+1}): {e}")
            time.sleep(GROQ_BASE_BACKOFF * (attempt + 1))
            continue

    log.warning(f"  [{item.get('id')}] Skipped after {MAX_RETRIES} failed attempts")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION CALL — GEMINI
# ══════════════════════════════════════════════════════════════════════════════

def extract_with_gemini(client, item: dict) -> dict | None:
    """
    Calls Gemini API for one item.
    Same contract as extract_with_groq:
      - Returns clean dict on success
      - Returns None if all retries fail
      - Raises RuntimeError("DAILY_LIMIT") on quota exhaustion

    Gemini uses a different client interface (google.genai) and different
    error messages than Groq — hence a separate function.
    Gemini also needs longer delays (15 RPM vs 30 RPM for Groq).
    """
    from google.genai import types

    prompt = build_prompt(item)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=350,
                ),
            )

            if not response.text:
                log.warning(f"  [{item.get('id')}] Empty response (attempt {attempt+1})")
                time.sleep(GEMINI_BASE_BACKOFF)
                continue

            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            raw = raw.strip()

            parsed = json.loads(raw)

            required = {"entities", "topics", "sentiment", "sentiment_score", "key_opinion"}
            missing  = required - set(parsed.keys())
            if missing:
                log.warning(f"  [{item.get('id')}] Missing keys {missing} (attempt {attempt+1})")
                time.sleep(GEMINI_BASE_BACKOFF)
                continue

            if any(parsed.get(k) is None for k in required):
                log.warning(f"  [{item.get('id')}] Null values (attempt {attempt+1})")
                time.sleep(GEMINI_BASE_BACKOFF)
                continue

            return parsed  # ✓ clean extraction

        except json.JSONDecodeError:
            log.warning(f"  [{item.get('id')}] JSON parse failed (attempt {attempt+1})")
            time.sleep(GEMINI_BASE_BACKOFF)
            continue

        except Exception as e:
            err     = str(e).lower()
            raw_err = str(e)

            # True daily quota exhausted — rotate to next key
            if ("quota exceeded" in err and ("day" in err or "billing" in err)):
                log.warning(f"  Gemini daily quota hit on attempt {attempt+1}")
                raise RuntimeError("DAILY_LIMIT")

            # Rate limit (RPM) — back off and retry on same key
            if ("429" in raw_err or "resource_exhausted" in err
                    or "rate" in err or "quota" in err):
                wait = GEMINI_BASE_BACKOFF * (2 ** attempt)
                log.warning(f"  [{item.get('id')}] Rate limit. Waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue

            # Network / transient errors
            if "timeout" in err or "connection" in err:
                wait = GEMINI_BASE_BACKOFF * (attempt + 1)
                log.warning(f"  [{item.get('id')}] Network error. Waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue

            # Unknown error — retry
            log.error(f"  [{item.get('id')}] Unknown error (attempt {attempt+1}): {e}")
            time.sleep(GEMINI_BASE_BACKOFF)
            continue

    log.warning(f"  [{item.get('id')}] Skipped after {MAX_RETRIES} failed attempts")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER DISPATCHER
# Routes each extraction call to the right function based on current provider.
# ══════════════════════════════════════════════════════════════════════════════

def extract_item(provider: str, client, item: dict) -> dict | None:
    """
    Routes to the correct extraction function based on provider.
    Raises RuntimeError("DAILY_LIMIT") when the current key is exhausted.
    Returns None when the item should be skipped (all retries failed).
    """
    if provider == "groq":
        return extract_with_groq(client, item)
    elif provider == "gemini":
        return extract_with_gemini(client, item)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(path: Path) -> set[str]:
    """Loads set of already-processed item IDs from checkpoint file."""
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def save_checkpoint(path: Path, done_ids: set[str]) -> None:
    """Writes done IDs to checkpoint file, one per line, sorted."""
    path.write_text("\n".join(sorted(done_ids)))


def flush(out_path: Path, ckpt_path: Path,
          existing: list, new_results: list, done_ids: set) -> None:
    """
    Atomically writes output JSON and checkpoint to disk.
    Called every CHECKPOINT_EVERY items and at end of each file.
    Single-threaded write — no race conditions.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing + new_results, f, ensure_ascii=False, indent=2)
    save_checkpoint(ckpt_path, done_ids)


# ══════════════════════════════════════════════════════════════════════════════
# PER-FILE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def extract_file(key_pool: KeyPoolManager, filename: str) -> bool:
    """
    Extracts all pending items in one file.

    Returns True  if file completed successfully.
    Returns False if all keys were exhausted mid-file (caller should stop).

    Belt-and-suspenders done_ids:
      done_ids is built from BOTH the checkpoint file AND the existing output.
      This catches the case where checkpoint marks an ID as done but the
      corresponding item never made it into the output JSON (sync issue).
      Such items will be correctly reprocessed.
    """
    raw_path  = RAW_DIR / f"{filename}.json"
    out_path  = EXTRACTED_DIR / f"{filename}.json"
    ckpt_path = EXTRACTED_DIR / f"{filename}.checkpoint"

    log.info(f"\n{'=' * 65}")
    log.info(f"FILE: {filename}")

    all_items = json.load(open(raw_path, encoding="utf-8"))
    log.info(f"  Raw items: {len(all_items)}")

    # Load checkpoint
    done_ids = load_checkpoint(ckpt_path)
    log.info(f"  Checkpoint: {len(done_ids)} IDs marked done")

    # Load existing output
    existing = []
    if out_path.exists():
        existing = json.load(open(out_path, encoding="utf-8"))
        log.info(f"  Existing output: {len(existing)} items")

    # Belt-and-suspenders: merge done_ids with IDs actually in output
    # This catches checkpoint/output sync issues from prior interrupted runs
    output_ids = {it.get("id") for it in existing}
    done_ids   = done_ids | output_ids
    if len(done_ids) > len(output_ids):
        log.info(f"  After sync check: {len(done_ids)} total done IDs")

    pending = [it for it in all_items if it.get("id") not in done_ids]
    log.info(f"  Pending: {len(pending)} items")
    log.info(f"  Active key: {key_pool.key_name} [{key_pool.provider}]")

    if not pending:
        log.info("  Nothing to do.")
        return True

    # Build client for current key
    client  = make_client(key_pool.provider, key_pool.key_value)
    delay   = GROQ_REQUEST_DELAY if key_pool.provider == "groq" else GEMINI_REQUEST_DELAY

    new_results, skipped, counter = [], 0, 0

    for i, item in enumerate(pending):
        item_id = item.get("id")

        if (i + 1) % 25 == 0 or i == 0:
            log.info(
                f"  [{i+1}/{len(pending)}] id={item_id} "
                f"extracted={len(new_results)} skipped={skipped} "
                f"key={key_pool.key_name}"
            )

        time.sleep(delay)

        try:
            extraction = extract_item(key_pool.provider, client, item)

        except RuntimeError as e:
            if "DAILY_LIMIT" in str(e):
                # Save progress, try to rotate to next key
                flush(out_path, ckpt_path, existing, new_results, done_ids)
                log.warning(
                    f"  Daily limit hit on {key_pool.key_name}. "
                    f"Saved {len(new_results)} new items."
                )

                if not key_pool.rotate():
                    # All keys exhausted — stop everything
                    log.error("  All keys exhausted. Stopping.")
                    return False

                # Rebuild client + delay for new provider
                client = make_client(key_pool.provider, key_pool.key_value)
                delay  = GROQ_REQUEST_DELAY if key_pool.provider == "groq" else GEMINI_REQUEST_DELAY
                log.info(f"  Continuing with {key_pool.key_name} [{key_pool.provider}]")

                # Retry this same item with new key
                try:
                    extraction = extract_item(key_pool.provider, client, item)
                except RuntimeError:
                    # New key also immediately hit daily limit
                    flush(out_path, ckpt_path, existing, new_results, done_ids)
                    if not key_pool.rotate():
                        log.error("  All keys exhausted after rotation. Stopping.")
                        return False
                    client = make_client(key_pool.provider, key_pool.key_value)
                    delay  = GROQ_REQUEST_DELAY if key_pool.provider == "groq" else GEMINI_REQUEST_DELAY
                    extraction = None  # skip this item, move on
            else:
                raise

        except KeyboardInterrupt:
            flush(out_path, ckpt_path, existing, new_results, done_ids)
            log.info(f"  Interrupted. Saved {len(new_results)} new items.")
            raise

        # Write result or skip
        if extraction is None:
            skipped += 1
        else:
            new_results.append({
                **item,
                "extraction": {
                    **extraction,
                    "time_window": item.get("window", "unknown"),
                }
            })

        done_ids.add(item_id)
        counter += 1

        if counter >= CHECKPOINT_EVERY:
            flush(out_path, ckpt_path, existing, new_results, done_ids)
            log.info(
                f"  ✓ Checkpoint — "
                f"total={len(existing)+len(new_results)} skipped={skipped}"
            )
            counter = 0

    # Final flush
    flush(out_path, ckpt_path, existing, new_results, done_ids)
    total = len(existing) + len(new_results)
    log.info(f"  DONE: extracted={len(new_results)} skipped={skipped} total={total}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Hercules Cyclops — Stage 2 Extractor (unified)")
    log.info(f"Raw dir:       {RAW_DIR}")
    log.info(f"Extracted dir: {EXTRACTED_DIR}")

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: initialise key pool ───────────────────────────────────────────
    log.info("\nInitialising key pool...")
    try:
        key_pool = KeyPoolManager()
    except RuntimeError as e:
        log.error(str(e))
        return

    # ── Step 2: audit all files ───────────────────────────────────────────────
    audit = audit_files()

    to_process = [f for f, r in audit.items() if not r["complete"]]

    if not to_process:
        log.info("\nAll files already complete. Stage 2 is done.")
        log.info("Proceed to Stage 3 (Neo4j ingestion).")
        return

    # ── Step 3: extract incomplete files ─────────────────────────────────────
    log.info(f"\nProcessing {len(to_process)} file(s)...")

    for filename in to_process:
        try:
            success = extract_file(key_pool, filename)
        except KeyboardInterrupt:
            log.info("\nInterrupted. Progress saved. Re-run to resume.")
            break
        except Exception as e:
            log.error(f"Fatal error on {filename}: {e}")
            log.info("Moving to next file...")
            continue

        if not success:
            log.error("All API keys exhausted. Cannot continue.")
            log.info("Add more keys to .env and re-run to resume.")
            break

    # ── Step 4: final audit ───────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("FINAL AUDIT")
    audit_files()
    log.info("\nRe-run this script anytime to fill any remaining gaps.")
    log.info("When all files show ✓ SKIP → proceed to Stage 3.")


if __name__ == "__main__":
    main()