"""
reddit_scraper.py
Scrapes Reddit data via Arctic Shift API (arctic-shift.photon-reddit.com).

WHY ARCTIC SHIFT (not PRAW or .json endpoints):
- Reddit OAuth (PRAW) closed to new accounts via Responsible Builder Policy, Nov 2025.
  Account age + karma required; new accounts are silently rejected.
- Reddit public .json endpoints (reddit.com/r/sub.json) killed May 30, 2026 — 403 Forbidden.
- Arctic Shift: community-maintained Pushshift successor, indexed through Apr 2026,
  no API key required, native date-range filtering, returns identical fields to PRAW.

SUBREDDIT SELECTION RATIONALE:
- Fitness_India (257k members, "crazy activity"): primary source. India-specific gym
  culture, protein/supplement discussion, urban fitness habits. Highest signal density.
- india (2M+ members): general sub but fitness threads get major traction. Captures
  broader cultural attitudes toward gym culture bleeding into mainstream conversation.
- bangalore (large metro sub): urban gym culture, local gym reviews, city-specific
  habits. Bangalore is India's fastest-growing fitness market.
- hyderabad (large metro sub): same rationale. Tier-1 metro with rapidly expanding
  gym culture. Adds geographic diversity vs Bangalore-only data.
- fitness (17M members, largest fitness sub on Reddit): global benchmark. Lets us
  compare Indian gym trends vs global conversation — critical for market research framing.

DATA STRATEGY:
- 5 subreddits across 3 time windows = 15 scrape jobs per run
- For general subs (india, bangalore, hyderabad): keyword filter applied to posts
- For topic-native subs (Fitness_India, fitness): all posts pass filter
- For each post: fetch full comment tree (parent_id chain preserved for graph edges)
- Output: raw JSON files per window per subreddit → data/raw/

OUTPUT FILES (15 total):
    window_1_recent_Fitness_India.json     (0–30 days)
    window_1_recent_india.json
    ... etc (3 windows × 5 subreddits)
"""

import json
import time
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_URL = "https://arctic-shift.photon-reddit.com/api"

# Gym/fitness in urban India — keyword set for filtering general subreddits
# Topic-native subs (Fitness_India, fitness) bypass this filter entirely
SEARCH_KEYWORDS = [
    # Core gym terms
    "gym", "fitness", "workout", "exercise", "training",
    # Nutrition / supplements
    "protein", "supplement", "whey", "creatine", "pre-workout",
    "mass gainer", "bcaa", "nutrition", "diet", "calories",
    # Goals
    "weight loss", "fat loss", "muscle", "bulking", "cutting",
    "body transformation", "weight gain", "abs", "shred",
    # Disciplines
    "bodybuilding", "cardio", "crossfit", "yoga", "running",
    "powerlifting", "calisthenics", "zumba",
    # Urban India specific
    "personal trainer", "gym membership", "cult fit", "fittr",
    "gold's gym", "anytime fitness", "f45",
]

# Ordered by expected data volume and signal quality
# Fitness_India first — it's the primary source
SUBREDDITS = [
    "Fitness_India",   # 257k members, highest India-fitness signal density
    "india",           # 2M+ members, mainstream fitness conversation
    "bangalore",       # Metro gym culture, local reviews
    "hyderabad",       # Metro gym culture, geographic diversity
    "fitness",         # 17M members, global benchmark for comparison
]

# Time windows: (label, days_start, days_end)
# days_start = how many days ago the window ENDS (closer to now)
# days_end   = how many days ago the window STARTS (further from now)
TIME_WINDOWS = [
    ("window_1_recent",     0,   30),
    ("window_2_midterm",   31,   90),
    ("window_3_historical", 91, 180),
]

# Arctic Shift: "be considerate" — their words. 1.5s between requests is safe.
REQUEST_DELAY_SECONDS = 1.5
MAX_POSTS_PER_WINDOW_PER_SUB = 200   # hard cap; 100 per page × 2 pages max
MAX_COMMENTS_PER_POST = 100          # top 100 comments per post
POSTS_PAGE_SIZE = 100                # Arctic Shift max limit per request
COMMENTS_PAGE_SIZE = 100

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


# ── HTTP client ────────────────────────────────────────────────────────────────
class ArcticShiftClient:
    """
    Thin wrapper around requests for Arctic Shift API.
    Handles:
      - User-Agent header (required by all Reddit-adjacent APIs)
      - Rate limit backoff via X-RateLimit-Remaining header
      - Retry on 429 / 5xx
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            # Format: <platform>:<app_name>:<version> (by /u/<your_reddit_username>)
            # Arctic Shift asks to be "considerate" — identify yourself clearly
            "User-Agent": "HerculesCyclops/1.0 (GraphRAG research tool; gym-fitness-india)",
            "Accept": "application/json",
        })

    def get(self, endpoint: str, params: dict, retries: int = 8) -> dict:
        """
        GET request with:
        - Rate limit header inspection
        - Exponential backoff on 429 / 5xx
        - Network drop recovery: waits up to 5 minutes for internet to return
        - Returns parsed JSON dict, raises only after all retries exhausted
        """
        url = f"{BASE_URL}{endpoint}"

        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)

                # Check rate limit header — pause if headroom is low
                remaining = response.headers.get("X-RateLimit-Remaining")
                if remaining is not None and int(remaining) < 5:
                    reset_ts = response.headers.get("X-RateLimit-Reset", 0)
                    wait = max(0, int(reset_ts) - int(time.time())) + 1
                    log.warning(f"Rate limit low ({remaining} remaining). Sleeping {wait}s.")
                    time.sleep(wait)

                if response.status_code == 429:
                    wait = 10 * (attempt + 1)
                    log.warning(f"429 received. Backing off {wait}s (attempt {attempt+1}).")
                    time.sleep(wait)
                    continue

                if response.status_code >= 500:
                    wait = 5 * (attempt + 1)
                    log.warning(f"Server error {response.status_code}. Retrying in {wait}s.")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                return response.json()

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                # Network drop — could be temporary (wifi hiccup, router reset)
                # Wait progressively longer before retrying: 30s, 60s, 90s...
                wait = 30 * (attempt + 1)
                log.warning(
                    f"Network error (attempt {attempt+1}/{retries}). "
                    f"Internet may be down. Waiting {wait}s before retry..."
                )
                time.sleep(wait)

            except requests.exceptions.RequestException as e:
                # Non-network error (bad response, etc) — shorter backoff
                log.error(f"Request failed (attempt {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(5 * (attempt + 1))

        raise RuntimeError(f"All {retries} retries failed for {url}")


# ── Time window calculation ────────────────────────────────────────────────────
def get_window_bounds(days_start: int, days_end: int) -> tuple[str, str]:
    """
    Returns (after_str, before_str) as ISO date strings for Arctic Shift.
    Relative to UTC now at runtime — so every run uses today's date.

    Example: days_start=0, days_end=30 → last 30 days
    """
    now = datetime.now(timezone.utc)
    before_dt = now - timedelta(days=days_start)
    after_dt  = now - timedelta(days=days_end)

    # Arctic Shift accepts YYYY-MM-DD or Unix timestamps
    return (
        after_dt.strftime("%Y-%m-%d"),
        before_dt.strftime("%Y-%m-%d"),
    )


# ── Post fetching ──────────────────────────────────────────────────────────────
def fetch_posts(
    client: ArcticShiftClient,
    subreddit: str,
    after: str,
    before: str,
    max_posts: int,
) -> list[dict]:
    """
    Fetches posts from a subreddit within [after, before] date range.
    Paginates using created_utc of the last result as the next 'before' cursor.

    Arctic Shift pagination model:
    - No cursor token — use before=<created_utc of last post> to get next page
    - Each page returns up to `limit` results sorted by created_utc desc
    - Stop when: result count < page_size (last page) OR max_posts reached
    """
    all_posts = []
    current_before = before

    log.info(f"  Fetching posts: r/{subreddit} | {after} → {before}")

    while len(all_posts) < max_posts:
        params = {
            "subreddit": subreddit,
            "after": after,
            "before": current_before,
            "limit": POSTS_PAGE_SIZE,
            # Arctic Shift /posts/search does not accept sort/order params
            # Results returned in created_utc desc by default
        }

        time.sleep(REQUEST_DELAY_SECONDS)
        data = client.get("/posts/search", params)
        page = data.get("data", [])

        if not page:
            break

        all_posts.extend(page)
        log.info(f"    Got {len(page)} posts (total: {len(all_posts)})")

        # Last page condition
        if len(page) < POSTS_PAGE_SIZE:
            break

        # Advance cursor: set before to created_utc of oldest post in this page
        # created_utc is a Unix timestamp (int) — convert to date string for comparison
        oldest_utc = page[-1].get("created_utc")
        if oldest_utc is None:
            break
        oldest_date = datetime.fromtimestamp(int(oldest_utc), tz=timezone.utc).strftime("%Y-%m-%d")
        if oldest_date <= after:
            break
        current_before = oldest_date

    return all_posts[:max_posts]


# ── Comment fetching ───────────────────────────────────────────────────────────
def fetch_comments_for_post(
    client: ArcticShiftClient,
    post_id: str,
) -> list[dict]:
    """
    Fetches all comments for a given post ID.
    Arctic Shift endpoint: /api/comments/search?link_id=<post_id>

    Returns flat list of comment dicts with parent_id field — lets us
    reconstruct the tree in graph_ingestor.py via:
        - parent_id starts with "t3_" → direct reply to post
        - parent_id starts with "t1_" → reply to another comment
    """
    all_comments = []
    # Arctic Shift uses t3_<id> format for link_id
    link_id = f"t3_{post_id}" if not post_id.startswith("t3_") else post_id

    params = {
        "link_id": link_id,
        "limit": COMMENTS_PAGE_SIZE,
        # Arctic Shift /comments/search does not accept sort/order params
    }

    time.sleep(REQUEST_DELAY_SECONDS)

    try:
        data = client.get("/comments/search", params)
        all_comments = data.get("data", [])
    except Exception as e:
        log.warning(f"    Failed to fetch comments for post {post_id}: {e}")

    return all_comments


# ── Keyword filtering ──────────────────────────────────────────────────────────
def is_fitness_relevant(post: dict) -> bool:
    """
    Filter posts to gym/fitness domain.
    Checks title + selftext against SEARCH_KEYWORDS (case-insensitive).

    Topic-native subs bypass filter entirely — everything there is on-topic:
      - Fitness_India: 257k members, all fitness discussion
      - fitness: 17M members, all fitness discussion

    General subs require at least one keyword match:
      - india, bangalore, hyderabad: large general communities where
        fitness is one of many topics; strict filter needed for signal quality
    """
    subreddit = post.get("subreddit", "").lower()

    # Topic-native subreddits — skip filtering
    if subreddit in ("fitness_india", "fitness"):
        return True

    # General subreddits — keyword gate
    title    = (post.get("title", "") or "").lower()
    selftext = (post.get("selftext", "") or "").lower()
    combined = title + " " + selftext

    return any(kw in combined for kw in SEARCH_KEYWORDS)


# ── Post normalisation ─────────────────────────────────────────────────────────
def normalise_post(post: dict, window_label: str) -> dict:
    """
    Extracts and renames only the fields we need downstream.
    Everything else is dropped — keeps JSON files lean.

    Fields kept and why:
      id            → graph node ID, comment link_id key
      title         → text for embedding + entity extraction
      selftext      → body text
      author        → Author node in graph
      subreddit     → Subreddit node in graph
      score         → engagement signal (edge weight)
      num_comments  → engagement signal
      created_utc   → CRITICAL: temporal graph timestamp
      url           → provenance / citation
      permalink     → full Reddit URL for source citation
      window        → which time window this came from (our metadata)
    """
    return {
        "id":           post.get("id"),
        "title":        post.get("title", ""),
        "selftext":     post.get("selftext", "") or "",
        "author":       post.get("author", "[deleted]"),
        "subreddit":    post.get("subreddit", ""),
        "score":        post.get("score", 0),
        "num_comments": post.get("num_comments", 0),
        "created_utc":  post.get("created_utc"),
        "url":          post.get("url", ""),
        "permalink":    f"https://reddit.com{post.get('permalink', '')}",
        "window":       window_label,
        "type":         "post",
    }


def normalise_comment(comment: dict, post_id: str, window_label: str) -> dict:
    """
    parent_id decoding:
      "t3_abc123" → direct reply to post abc123
      "t1_xyz789" → reply to comment xyz789
    We store both raw parent_id and a clean parent_type field.
    graph_ingestor.py will use this to build REPLIED_TO edges.
    """
    parent_id  = comment.get("parent_id", "")
    parent_type = "post" if parent_id.startswith("t3_") else "comment"

    return {
        "id":          comment.get("id"),
        "body":        comment.get("body", "") or "",
        "author":      comment.get("author", "[deleted]"),
        "subreddit":   comment.get("subreddit", ""),
        "score":       comment.get("score", 0),
        "created_utc": comment.get("created_utc"),
        "post_id":     post_id,
        "parent_id":   parent_id,
        "parent_type": parent_type,
        "permalink":   f"https://reddit.com{comment.get('permalink', '')}",
        "window":      window_label,
        "type":        "comment",
    }


# ── Main scrape orchestrator ───────────────────────────────────────────────────
def scrape_window(
    client: ArcticShiftClient,
    window_label: str,
    days_start: int,
    days_end: int,
) -> dict[str, list[dict]]:
    """
    Scrapes all subreddits for one time window.
    Returns dict keyed by subreddit → list of normalised post+comment dicts.
    """
    after, before = get_window_bounds(days_start, days_end)
    log.info(f"\n{'='*60}")
    log.info(f"Window: {window_label} | {after} → {before}")
    log.info(f"{'='*60}")

    window_results = {}

    for subreddit in SUBREDDITS:
        log.info(f"\nSubreddit: r/{subreddit}")
        sub_data = []

        raw_posts = fetch_posts(client, subreddit, after, before, MAX_POSTS_PER_WINDOW_PER_SUB)

        # Filter for fitness relevance (only strict for general subs)
        relevant_posts = [p for p in raw_posts if is_fitness_relevant(p)]
        log.info(f"  Fitness-relevant posts: {len(relevant_posts)}/{len(raw_posts)}")

        for i, post in enumerate(relevant_posts):
            post_id = post.get("id")
            if not post_id:
                continue

            norm_post = normalise_post(post, window_label)
            sub_data.append(norm_post)

            # Fetch comments for this post
            log.info(f"  [{i+1}/{len(relevant_posts)}] Fetching comments for post: {post_id}")
            raw_comments = fetch_comments_for_post(client, post_id)

            for comment in raw_comments:
                # Skip deleted/removed comments — no extractable signal
                body = (comment.get("body", "") or "").strip()
                if body in ("[deleted]", "[removed]", ""):
                    continue
                sub_data.append(normalise_comment(comment, post_id, window_label))

            log.info(f"    Post {post_id}: {len(raw_comments)} comments")

        window_results[subreddit] = sub_data
        log.info(f"  r/{subreddit} total items: {len(sub_data)}")

    return window_results


def save_results(window_label: str, results: dict[str, list[dict]]) -> None:
    """
    Saves one JSON file per subreddit per window.
    File format: {window_label}_{subreddit}.json
    Each file is a JSON array of normalised post/comment dicts.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for subreddit, items in results.items():
        filename = OUTPUT_DIR / f"{window_label}_{subreddit}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        log.info(f"Saved {len(items)} items → {filename}")


def scrape_all() -> None:
    """
    Entry point. Runs all 3 windows × 4 subreddits.
    Prints a summary at the end.
    """
    log.info("Hercules Cyclops — Reddit Scraper")
    log.info(f"Run date (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    log.info(f"Subreddits: {SUBREDDITS}")
    log.info(f"Windows: {[(w[0], f'{w[1]}–{w[2]} days ago') for w in TIME_WINDOWS]}")

    client = ArcticShiftClient()
    summary = {}

    for window_label, days_start, days_end in TIME_WINDOWS:
        results = scrape_window(client, window_label, days_start, days_end)
        save_results(window_label, results)

        # Track totals for summary
        for sub, items in results.items():
            posts    = sum(1 for i in items if i["type"] == "post")
            comments = sum(1 for i in items if i["type"] == "comment")
            key = f"{window_label}/{sub}"
            summary[key] = {"posts": posts, "comments": comments, "total": len(items)}

    # Final summary table
    log.info("\n" + "="*60)
    log.info("SCRAPE COMPLETE — Summary")
    log.info("="*60)
    total_posts = total_comments = 0
    for key, counts in summary.items():
        log.info(f"  {key:<40} posts={counts['posts']:>4}  comments={counts['comments']:>5}")
        total_posts    += counts["posts"]
        total_comments += counts["comments"]
    log.info("-"*60)
    log.info(f"  TOTAL  posts={total_posts}  comments={total_comments}  items={total_posts+total_comments}")
    log.info(f"  Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    scrape_all()