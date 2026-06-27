"""
test_scraper.py
Run this BEFORE the full scrape to confirm Arctic Shift is reachable
and returning data for our target subreddits.

Usage:
    python test_scraper.py

Expected output:
    ✅ Fitness_India  — X posts returned
    ✅ india          — X posts returned
    ✅ bangalore      — X posts returned
    ✅ hyderabad      — X posts returned
    ✅ fitness        — X posts returned
    All good — run the full scraper.
"""

import requests
import json
from datetime import datetime, timezone, timedelta

BASE_URL = "https://arctic-shift.photon-reddit.com/api"
SUBREDDITS = ["Fitness_India", "india", "bangalore", "hyderabad", "fitness"]

# Test window: last 30 days
now = datetime.now(timezone.utc)
after  = (now - timedelta(days=30)).strftime("%Y-%m-%d")
before = now.strftime("%Y-%m-%d")

headers = {
    "User-Agent": "HerculesCyclops/1.0 (GraphRAG research; gym-fitness-india)",
    "Accept": "application/json",
}

print(f"\nArctic Shift connectivity test")
print(f"Window: {after} → {before}\n")

all_ok = True

for sub in SUBREDDITS:
    try:
        resp = requests.get(
            f"{BASE_URL}/posts/search",
            params={
                "subreddit": sub,
                "after": after,
                "before": before,
                "limit": 3,   # just 3 posts to confirm it works
            },
            headers=headers,
            timeout=15,
        )

        if resp.status_code == 200:
            data = resp.json()
            posts = data.get("data", [])
            if posts:
                sample_title = posts[0].get("title", "")[:60]
                print(f"  ✅ r/{sub:<16} — {len(posts)} posts  |  sample: \"{sample_title}\"")
            else:
                print(f"  ⚠️  r/{sub:<16} — 0 posts returned (subreddit may be sparse in this window)")
                all_ok = False
        else:
            print(f"  ❌ r/{sub:<16} — HTTP {resp.status_code}: {resp.text[:100]}")
            all_ok = False

    except requests.exceptions.ConnectionError:
        print(f"  ❌ r/{sub:<16} — Connection failed. Check internet connection.")
        all_ok = False
    except requests.exceptions.Timeout:
        print(f"  ❌ r/{sub:<16} — Timeout. Arctic Shift may be slow, try again.")
        all_ok = False
    except Exception as e:
        print(f"  ❌ r/{sub:<16} — Unexpected error: {e}")
        all_ok = False

    import time
    time.sleep(1.5)  # be polite even during testing

print()
if all_ok:
    print("✅ All subreddits reachable. Run the full scraper:")
    print("   python -m src.ingestion.reddit_scraper")
else:
    print("⚠️  Some subreddits had issues. Check above before running full scrape.")

# Also test comments endpoint for one post
print("\nTesting comments endpoint...")
try:
    resp = requests.get(
        f"{BASE_URL}/posts/search",
        params={"subreddit": "Fitness_India", "after": after, "before": before, "limit": 1},
        headers=headers,
        timeout=15,
    )
    posts = resp.json().get("data", [])
    if posts:
        post_id = posts[0]["id"]
        resp2 = requests.get(
            f"{BASE_URL}/comments/search",
            params={"link_id": f"t3_{post_id}", "limit": 3},
            headers=headers,
            timeout=15,
        )
        comments = resp2.json().get("data", [])
        print(f"  ✅ Comments endpoint — {len(comments)} comments for post {post_id}")
        if comments:
            print(f"     Sample parent_id: {comments[0].get('parent_id', 'N/A')}")
    else:
        print("  ⚠️  No posts to test comments endpoint with")
except Exception as e:
    print(f"  ❌ Comments endpoint failed: {e}")

print()