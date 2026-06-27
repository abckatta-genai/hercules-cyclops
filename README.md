# Hercules Cyclops — Hybrid GraphRAG System

**Assignment:** GenAI Engineer, Jupiter Meta Labs  
**Topic:** Gym culture and fitness habits in urban India  
**Built by:** Ashish Katta

---

## Table of Contents

1. [The Problem](#the-problem)
2. [What This System Does](#what-this-system-does)
3. [Data Design — Topic, Subreddits, and Time Windows](#data-design)
4. [Pipeline — Stage by Stage](#pipeline)
5. [Tech Stack — Choices and Reasoning](#tech-stack)
6. [Demo Output — Real Results](#demo-output)
7. [Setup](#setup)
8. [Project Structure](#project-structure)
9. [Module Status](#module-status)

---

## The Problem

Consumer intelligence platforms face a fundamental challenge: **surveys are slow, expensive, and biased**. By the time a brand commissions a survey about gym supplement sentiment, designs it, distributes it, and analyses the results — the market has already moved.

People are already talking. Every day, thousands of gym-goers in urban India post about what supplements they use, what brands disappointed them, what they can no longer afford, what they switched to. This opinion data is public, real-time, and unprompted — which makes it far more honest than a survey response.

The harder problem is **temporal intelligence**: not just what people think, but how their thinking changed over time. A brand doesn't just want to know that MuscleBlaze sentiment is negative. They want to know that it was positive six months ago, turned negative in March when prices spiked 4X, and is now producing active brand-switching behaviour toward Nakpro and OWN.

That is the problem Hercules Cyclops solves.

It ingests Reddit at scale, structures the data as a temporal knowledge graph, maintains a parallel vector index for semantic search, and answers natural language questions about how opinions evolved — with citations pointing back to the original posts.

This mirrors the core intelligence challenge at Jupiter Meta Labs' Hercules platform: ingest opinion data at scale, model entities and relationships over time, and surface insights that would otherwise require weeks of manual analyst work.

---

## What This System Does

Hercules Cyclops answers market research questions about how public opinion evolves over time. It reads what people are already saying on Reddit, structures it, and produces cited answers to questions like:

> *"How has sentiment around whey protein prices changed in the last 6 months?"*  
> *"Which supplement brands are most discussed, and what do people actually think of them?"*  
> *"What do Indian gym-goers recommend for muscle building on a budget?"*  
> *"What fitness concerns emerged recently that weren't discussed 6 months ago?"*

It does this using a **Hybrid GraphRAG pipeline** — combining a knowledge graph (entities, relationships, and when things happened) with semantic vector search (meaning-based retrieval), fused together for answers that neither system could produce alone.

### Architecture

```
Reddit (Arctic Shift Archive)
           │
           ▼
┌──────────────────────┐
│  reddit_scraper.py   │  ← Stage 1: Raw data collection
│  (Arctic Shift API)  │    5 subreddits scraped × 3 time windows
└────────┬─────────────┘
         │ 11,536 items → data/raw/ (15 JSON files, 3 subreddits extracted)
         ▼
┌──────────────────────┐
│    extractor.py      │  ← Stage 2: LLM extraction
│ (Groq llama-3.1-8b) │    entities, topics, sentiment per item
└────────┬─────────────┘
         │ 11,048 enriched items → data/extracted/ (9 JSON files)
         ▼
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌──────────┐
│ Neo4j │ │ ChromaDB │  ← Stage 3 + 4: Dual storage
│(graph)│ │ (vector) │    same data, two representations
└───┬───┘ └────┬─────┘
    │           │
    ▼           ▼
┌──────────┐ ┌──────────────┐
│ graph_   │ │  vector_     │  ← Stage 5: Parallel retrieval
│retriever │ │  retriever   │
└────┬─────┘ └──────┬───────┘
     │               │
     └───────┬───────┘
             ▼
      ┌─────────────┐
      │   rrf.py    │  ← Stage 5: Reciprocal Rank Fusion
      │  (fusion)   │    merges graph + vector results
      └──────┬──────┘
             ▼
     ┌──────────────────┐
     │ answer_generator │  ← Stage 6: Answer generation
     │  (Claude Sonnet) │    cited, temporal-aware answers
     └──────────────────┘
```

---

## Data Design

### Why Gym Culture in Urban India

Gym culture in urban India is one of the fastest-moving consumer segments in the country right now. Protein supplement penetration is growing rapidly, new gym chains are expanding into Tier-1 and Tier-2 cities, and pricing pressures (a 4X whey cost surge hit in early 2026) are actively reshaping brand loyalty. This makes it a high-signal, fast-moving topic — exactly the kind of market where temporal intelligence has real commercial value.

For a consumer intelligence platform, fitness is also a rich domain: strong opinions, named brands, measurable sentiment, clear temporal events (price changes, product launches, scandals), and an engaged online community that talks openly. Every post is a data point.

### Subreddit Selection — Why These Five

Five subreddits were chosen to cover three distinct dimensions of the topic:

| Subreddit | Members | Role | Why Chosen |
|---|---|---|---|
| r/Fitness_India | 257k | Primary source | India-specific gym culture, highest signal density for our topic. Supplement reviews, training discussions, urban fitness habits — all directly on-topic. |
| r/india | 2M+ | Cultural breadth | Mainstream Indian conversation. Fitness threads here represent mass-market sentiment, not the enthusiast bubble. |
| r/bangalore | Large | Metro specificity | Bangalore is India's fastest-growing fitness market. Local gym reviews, city-specific supplement availability, metro pricing. |
| r/hyderabad | Large | Geographic diversity | Second major metro. Adds geographic signal beyond Bangalore-only data. |
| r/fitness | 17M | Global benchmark | Largest fitness community on Reddit. Lets us compare Indian trends vs global conversation — critical for market research framing. |

Together these cover: **topic specificity** (Fitness_India), **geographic specificity** (bangalore, hyderabad), **cultural breadth** (india), and **global benchmark** (fitness).

### Why 9 Extracted Files, Not 15

The scraper collected data from all 5 subreddits across all 3 time windows — producing 15 raw JSON files in `data/raw/`. However, during extraction, **r/india and r/hyderabad were excluded** for two reasons:

**Signal quality:** r/india and r/hyderabad are large general communities where fitness is one of many topics. Despite keyword filtering during scraping, the extracted content from these subreddits had significantly lower fitness signal density — posts were about gym costs in passing rather than as the primary subject. Including them would dilute the quality of graph entities and vector embeddings.

**Data volume:** After keyword filtering, r/india and r/hyderabad contributed only 53–131 items per window — compared to 800–3,009 items per window from Fitness_India and fitness. Statistically, they would not meaningfully influence retrieval results while adding noise to entity deduplication.

The decision to extract 9 files (3 subreddits × 3 windows) over 15 is a deliberate **signal quality decision**, not a failure. The three retained subreddits — Fitness_India (primary), fitness (global benchmark), and bangalore (metro specificity) — cover the full analytical range needed for temporal market research.

### Time Windows

All windows are computed relative to the scraper run date — not hardcoded dates.

| Window | Range | Purpose |
|---|---|---|
| window_1_recent | Last 30 days | Current conversation — latest trends and sentiment |
| window_2_midterm | 31–90 days ago | Emerging patterns — things gaining or losing traction |
| window_3_historical | 91–180 days ago | Baseline — what conversation looked like 6 months ago |

This three-window design directly enables the assignment's core temporal queries. "How has X changed over 6 months" requires a before-state (window 3), a transition state (window 2), and a current state (window 1).

---

## Pipeline

### Stage 1 — Reddit Scraper (`src/ingestion/reddit_scraper.py`)

**What it does:** Scrapes Reddit posts and comments from 5 subreddits across 3 time windows using the Arctic Shift API.

**Input:** Subreddit names + time window bounds (computed at runtime relative to today)

**Output:** 15 JSON files in `data/raw/` — one per subreddit per window. Of these, 9 files (3 subreddits × 3 windows) proceed to extraction after signal quality filtering drops r/india and r/hyderabad.

**Key decisions:**
- **Arctic Shift over PRAW:** Reddit OAuth was closed to new accounts (Nov 2025 Responsible Builder Policy). Reddit's public `.json` endpoints were killed May 30, 2026. Arctic Shift is the only remaining free path to structured Reddit data at scale.
- **Pagination via `created_utc` cursor:** Arctic Shift has no token-based pagination — the scraper uses the `created_utc` of the last result as the `before` parameter for the next page.
- **Comment fetching per post:** For each post, all top-100 comments are fetched separately. `parent_id` is preserved (`t3_` = reply to post, `t1_` = reply to comment) for graph edge construction in Stage 3.
- **Keyword filter on general subreddits:** r/india, r/bangalore, r/hyderabad are large general communities. Posts are filtered against a 30-term keyword list (gym, protein, supplement, etc.) before saving. Topic-native subs (Fitness_India, fitness) bypass this filter.
- **Rate limiting:** 1.5s between requests, exponential backoff on 429s, `caffeinate -i` on macOS to prevent sleep during long runs.

**Actual numbers:**
- 1,278 posts + 10,258 comments = **11,536 items**
- 15 raw JSON files scraped (5 subreddits × 3 windows)
- 9 files proceed to extraction (r/india and r/hyderabad excluded on signal quality grounds)

---

### Stage 2 — LLM Extraction (`src/ingestion/extractor.py`)

**What it does:** Reads every raw item, calls an LLM to extract structured data, writes enriched JSON to `data/extracted/`.

**Input:** 15 raw JSON files from `data/raw/`

**Output:** 9 enriched JSON files in `data/extracted/` (3 subreddits × 3 windows — see data design section above)

**Extraction schema per item:**
```json
{
  "entities": [{"name": "MuscleBlaze", "type": "Brand"}],
  "topics": ["protein powder", "supplement pricing", "brand comparison"],
  "sentiment": "negative",
  "sentiment_score": -0.7,
  "key_opinion": "User reports MuscleBlaze prices have become unaffordable.",
  "time_window": "window_1_recent"
}
```

Entity types (9, fixed — map directly to Neo4j node labels in Stage 3):
`Person`, `Brand`, `Product`, `Exercise`, `Location`, `Supplement`, `Gym`, `Diet`, `Other`

**Key decisions:**
- **Groq llama-3.1-8b-instant over Claude/Gemini:** 11k+ LLM calls at extraction volume. Claude costs ~$3/M tokens — budget runs out fast. Gemini billing ran ₹858 of ₹1000 in a single partial run. Groq free tier: 500k tokens/day per key, zero cost. 6 keys used in parallel via automatic rotation.
- **Null policy:** If extraction fails after all retries, the item is **skipped entirely** — never written with null fields. Null values in Neo4j and ChromaDB cause cascade failures downstream. A few missing items are acceptable; corrupted data is not.
- **Belt-and-suspenders checkpointing:** Done IDs are built from both the checkpoint file AND the actual output file. This catches cases where the checkpoint marked an ID as done but the corresponding item never made it to the output JSON (a sync issue that occurred during multi-worker runs and was caught by this approach).
- **Automatic key rotation:** When a Groq key hits its daily limit, the script automatically rotates to the next available key (Groq keys 1–6, then Gemini as fallback) without stopping.
- **Single-threaded writes:** All file writes are single-threaded to eliminate race conditions. Prior parallel worker attempts created checkpoint/output sync issues — this design prevents that entirely.

**Actual numbers:**
- 11,536 raw items → **11,048 successfully extracted** (488 skipped after retry exhaustion)
- Zero null-field items in any output file
- 9 output files, ~15MB total

---

### Stage 3 — Neo4j Graph Ingestion (`src/graph/`)

**What it does:** Reads all 9 extracted JSON files and creates a property graph in Neo4j AuraDB.

**Input:** 9 enriched JSON files from `data/extracted/`

**Output:** Neo4j graph with 6 node types and 6 relationship types

**Graph schema:**

Nodes:
```
(Post)       — id, title, selftext, score, created_utc, window, subreddit, sentiment, sentiment_score
(Comment)    — id, body, score, created_utc, window, subreddit, sentiment, sentiment_score
(Author)     — name
(Entity)     — name, type  (e.g. "muscleblaze", "Brand")
(Topic)      — name        (e.g. "protein powder")
(Subreddit)  — name
```

Relationships:
```
(Author)  -[:POSTED]->    (Post)      author wrote this post
(Author)  -[:COMMENTED]-> (Comment)   author wrote this comment
(Comment) -[:ON]->        (Post)      comment belongs to this post
(Post)    -[:MENTIONS]->  (Entity)    post mentions this entity
(Post)    -[:TAGGED]->    (Topic)     post is tagged with this topic
(Post)    -[:IN]->        (Subreddit) post is in this subreddit
```

**Key decisions:**
- **MERGE not CREATE:** The same author appears across hundreds of posts. `MERGE` gives one Author node with hundreds of POSTED relationships — not hundreds of duplicate nodes. Same logic applies to Entity and Topic nodes. This is what makes the graph meaningful for traversal queries.
- **Constraints before ingestion:** Uniqueness constraints on Post.id, Comment.id, Author.name, Subreddit.name, and (Entity.name, Entity.type) are created before any data is written. Without constraints, concurrent MERGE calls can create duplicates.
- **Indexes on temporal fields:** `created_utc` and `window` are indexed on Post and Comment nodes. Retrieval queries always filter by these fields first.
- **Batch transactions:** Items are written in batches of 100 per transaction. One transaction per item = 11k round trips to AuraDB. One transaction for everything = catastrophic on failure. Batches balance speed and fault tolerance.
- **Entity normalisation:** Entity names are lowercased before storage. "MuscleBlaze" and "muscleblaze" merge to one node.

**Actual numbers:**
- **1,215** Post nodes
- **9,833** Comment nodes
- **4,750** Author nodes (deduplicated via MERGE)
- **5,953** Entity nodes (deduplicated by name + type)
- **727** Topic nodes
- **3** Subreddit nodes
- **54,588** total relationships

---

### Stage 4 — ChromaDB Vector Ingestion (`src/vector/`)

**What it does:** Embeds every item using `all-MiniLM-L6-v2` and stores the vectors in ChromaDB with metadata for filtered retrieval.

**Input:** 9 enriched JSON files from `data/extracted/`

**Output:** ChromaDB collection with 11,048 vectors + metadata

**Text to embed per item:**
- Post → `title + selftext + key_opinion` (full context + LLM summary)
- Comment → `body + key_opinion`

Appending `key_opinion` (the LLM's one-sentence summary of the author's main point) to the embedding text improves semantic search quality — it adds a clean signal alongside potentially noisy raw text.

**Metadata stored per vector** (enables pre-filtered search):
```
item_id, type, subreddit, window, author,
sentiment, sentiment_score, created_utc, permalink, entities, topics
```

**Key decisions:**
- **all-MiniLM-L6-v2:** Local model, no API calls, no cost, no rate limits. 384 dimensions, strong semantic performance for short-to-medium English text. Ran on Apple MPS (Metal GPU) — 11k embeddings completed in under 10 minutes.
- **Cosine similarity:** sentence-transformers produce unit-normalised vectors. Cosine similarity measures the angle between vectors — the correct metric for semantic similarity search.
- **Metadata filtering before search:** ChromaDB applies metadata filters before the ANN (approximate nearest neighbour) search. Filtering by `window` or `subreddit` first and then searching the subset is faster than searching all 11k vectors and post-filtering.
- **Upsert not insert:** `collection.upsert()` is idempotent — safe to re-run. No duplicate vectors created if the script is run multiple times.

**Actual numbers:**
- **11,048** vectors at 384 dimensions
- Cosine similarity space
- Embedding time: ~8 minutes on Apple MPS

---

### Stage 5 — Retrieval (`src/retrieval/`)

**What it does:** For a given query, runs graph retrieval and vector retrieval in parallel, then fuses the results.

#### Graph Retriever (`graph_retriever.py`)

Extracts keywords from the query (removing stop words), then runs two Cypher queries:

1. **Entity match** — finds posts/comments that `[:MENTIONS]` entities whose name contains any query keyword. Scores by mention count × engagement.
2. **Topic match** — finds posts `[:TAGGED]` with topics containing any query keyword. Broader than entity match — catches thematic relevance.

Results are merged and deduplicated by ID before returning.

**Why graph retrieval matters:** Graph traversal finds structural relationships that vector search cannot. "Which brands are most mentioned" requires counting `[:MENTIONS]` edges — a graph operation. "What topics appear in window_1 but not window_3" requires comparing node sets across time — another graph operation.

#### Vector Retriever (`vector_retriever.py`)

Embeds the query using the same `all-MiniLM-L6-v2` model used during ingestion (critical — query and documents must be in the same vector space), then runs cosine similarity search in ChromaDB.

Supports metadata pre-filtering: pass `window=`, `subreddit=`, or `sentiment=` to restrict the search space before the ANN query runs.

**Why vector retrieval matters:** Semantic search captures meaning regardless of exact wording. "Feeling burned out at the gym" matches posts about "lost the drive to lift" even with zero keyword overlap. Graph retrieval would miss these entirely.

#### Reciprocal Rank Fusion (`src/fusion/rrf.py`)

Takes the two ranked lists and merges them:

```
RRF_score(d) = Σ 1/(k + rank(d))    where k=60
```

Why RRF and not score averaging: graph scores (e.g. 120.0) and vector scores (e.g. 0.65) are on completely different scales. Averaging is meaningless. RRF only uses rank position — scale-agnostic by design.

Effect: documents appearing high in **both** lists get the biggest boost. Documents appearing in only one list rank lower. Items tagged `hybrid` (source appeared in both retrievers) consistently produced the most relevant final answers.

---

### Stage 6 — Answer Generation (`src/generation/answer_generator.py`)

**What it does:** Takes the top-10 fused results, formats them as numbered sources, and calls Claude Sonnet to generate a cited, temporal-aware answer.

**Input:** Query string + fused results from RRF

**Output:** Natural language answer with inline citations ([1], [2]...) and a sources section

**Prompt design:**
- Claude is given its role: market research analyst for a consumer intelligence platform
- The three time windows are explained explicitly
- Sources are formatted with window label, subreddit, and sentiment score
- Claude is instructed to cite inline, note which window findings come from, and acknowledge conflicts between sources
- Query type hints (`temporal`, `entity`, `sentiment`) adjust the instruction emphasis

**Why Claude Sonnet (not Groq) for generation:**
Answer generation requires nuanced multi-source synthesis, proper citation formatting, and temporal reasoning. Claude Sonnet outperforms smaller models on all of these. Groq/llama-3.1-8b was correct for extraction (schema-filling at 11k volume). Generation is a different task with different requirements.

**Budget conservation:** Claude Sonnet is called **only at query time** — never during ingestion. Each query uses approximately 1–2k tokens.

---

## Tech Stack — Choices and Reasoning

| Component | Choice | Rejected Alternatives | Reason |
|---|---|---|---|
| Reddit data | Arctic Shift API | PRAW (OAuth blocked Nov 2025), .json endpoints (killed May 2026) | Only remaining free path to Reddit data at scale |
| Extraction LLM | Groq llama-3.1-8b-instant | Claude (too expensive at 11k calls), Gemini (billing exhausted rapidly) | Free tier, 500k TPD per key, schema-filling doesn't need reasoning depth |
| Graph DB | Neo4j AuraDB Free | PostgreSQL, MongoDB | Native graph traversal, Cypher expressiveness, hosted with zero infra overhead |
| Vector DB | ChromaDB (local) | Pinecone, Weaviate | Zero infrastructure, Python-native, cloneable in under 10 minutes |
| Embeddings | all-MiniLM-L6-v2 | OpenAI text-embedding-3-small | Free, local, no rate limits, strong semantic performance for social media text |
| Fusion | Reciprocal Rank Fusion | Score averaging | Graph and vector scores are incomparable scales — RRF is rank-based and scale-agnostic |
| Generation LLM | Claude Sonnet | Groq llama-3.1-8b | Multi-source synthesis, citation formatting, temporal reasoning |
| Framework | Vanilla Python | LangChain, LlamaIndex | Hiring assignment — evaluators want to see deliberate engineering, not framework wiring |

---

## Demo Output — Real Results

These are actual outputs from running `demo.py` against the live system.

### Query 1 — Semantic: Gym Motivation and Burnout

*"What do people say about losing motivation to work out or feeling burned out at the gym?"*

The system identified that burnout is discussed quietly — surfacing in posts where users with visible progress (12kg lost) still admitted feeling like quitting. It distinguished between genuine motivational dropout and environmental factors (45°C summer heat disrupting gym timing — a distinctly urban Indian seasonal stressor). Sentiment showed a slight positive drift from historical to recent, but burnout acknowledgment became more explicit. r/Fitness (global) surfaced the starkest nihilism: *"There are people who feel like going to the gym?"*

### Query 2 — Graph: Supplement Brand Mentions

*"Which supplement brands are most frequently mentioned and what sentiment do they have?"*

OWN (Only What's Needed) led in visibility with mixed-to-positive sentiment. Big Muscle Whey carried the strongest negative signal (-1.00), explicitly called a scam across multiple posts. The system correctly identified that brand-specific discussions cooled in the recent window — a finding that requires temporal metadata on graph nodes and is invisible to pure semantic search.

### Query 3 — Hybrid: Budget Protein Recommendations

*"What protein supplements do Indian gym-goers recommend for muscle building on a budget?"*

The system detected an active brand-switching trend: MuscleBlaze was the historical go-to, but a 4X whey price surge drove users toward Nakpro, OWN, and whole-food protein stacking (Amul high-protein milk, Heritage paneer, eggs at ~₹215/day for 90g protein). The answer correctly surfaced a conflict between sources: one user argued plant proteins mixed with milk are illogical for lactose-intolerant individuals.

### Query 4 — Time-Series: Whey Protein Price Sentiment

*"How has sentiment around whey protein prices changed over the last 6 months?"*

Clear trajectory identified:

- **Historical (91–180 days ago):** Price anxiety absent. Users focused on hitting protein targets affordably — MuscleBlaze and ON recommended without concern.
- **Mid-term (31–90 days ago):** Crisis emerges. Economic Times report on 4X price surge enters discourse. Food Pharmer (influencer) explains global sourcing cost increases. Users begin brand-switching.
- **Recent (last 30 days):** Resignation hardens. Users lamenting when whey cost ₹1,500–₹1,800. One post: *"raw materials have become very expensive. Not sure if prices will ever come down."*

Trajectory: **neutral/positive → crisis awareness → entrenched negativity.**

---

## Setup

### Prerequisites

- Python 3.10+
- Neo4j AuraDB Free account ([console.neo4j.io](https://console.neo4j.io))
- Groq API key(s) ([console.groq.com](https://console.groq.com)) — free tier, no credit card
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))

### Quick Start (Pre-populated data — just run the demo)

If you have received this repo with `data/extracted/` and `data/chroma/` already populated:

```bash
# 1. Clone
git clone https://github.com/<your-username>/hercules-cyclops.git
cd hercules-cyclops

# 2. Install dependencies (~3 minutes)
pip install sentence-transformers==2.7.0 transformers==4.41.0 chromadb neo4j groq anthropic

# 3. Configure credentials
cp .env.example .env
# Edit .env — fill in NEO4J_URI, NEO4J_PASSWORD, ANTHROPIC_API_KEY

# 4. Export credentials
export $(grep -E 'NEO4J|ANTHROPIC' .env | xargs)

# 5. Run demo
python demo.py
```

Expected output: 4 queries with graph results, vector results, fused results, and Claude-generated cited answers. Takes approximately 3–5 minutes.

### Full Pipeline (From Scratch)

Run each stage in order. Each stage saves its output to disk — if interrupted, it resumes from where it left off.

```bash
# Stage 1: Scrape Reddit (~2–3 hours, runs unattended)
# Produces: data/raw/ — 15 JSON files (5 subs × 3 windows), ~11,500 items
# Note: only 9 files (3 subs × 3 windows) proceed to extraction
python src/ingestion/reddit_scraper.py

# Stage 2: LLM extraction (~6–8 hours with Groq free tier)
# Produces: data/extracted/ — 9 JSON files, ~11,000 items
# Requires: GROQ_API_KEY_1 through GROQ_API_KEY_6 in .env (use as many as you have)
export $(grep GROQ .env | xargs)
python src/ingestion/extractor.py

# Stage 3: Neo4j graph ingestion (~40 minutes)
# Produces: 11k nodes, 54k relationships in AuraDB
export $(grep NEO4J .env | xargs)
python src/graph/graph_ingestor.py

# Stage 4: ChromaDB vector ingestion (~10 minutes)
# Produces: data/chroma/ — 11k vectors
python src/vector/vector_ingestor.py

# Stages 5+6: Run demo
export $(grep -E 'NEO4J|ANTHROPIC' .env | xargs)
python demo.py
```

### Environment Variables

```bash
# .env.example

# Neo4j AuraDB
NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>

# Groq — add as many keys as you have (1 minimum, 6 for full parallel extraction)
GROQ_API_KEY_1=gsk_...
GROQ_API_KEY_2=gsk_...
GROQ_API_KEY_3=gsk_...
GROQ_API_KEY_4=gsk_...
GROQ_API_KEY_5=gsk_...
GROQ_API_KEY_6=gsk_...

# Gemini (optional — used as fallback if all Groq keys exhausted)
GEMINI_API_KEY=...

# Anthropic (for answer generation only)
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Project Structure

```
hercules-cyclops/
├── .env                            # credentials (not committed)
├── .env.example                    # documented template
├── requirements.txt
├── README.md
├── demo.py                         # end-to-end 4-query demonstration
├── data/
│   ├── raw/                        # Stage 1 output: 15 JSON files (5 subs × 3 windows scraped)
│   ├── extracted/                  # Stage 2 output: 9 enriched JSON files
│   └── chroma/                     # Stage 4 output: ChromaDB vector store
└── src/
    ├── ingestion/
    │   ├── reddit_scraper.py       # Arctic Shift scraper, 5 subs × 3 windows
    │   └── extractor.py           # Groq LLM extraction, 6-key pool, resumable
    ├── graph/
    │   ├── neo4j_client.py        # connection + schema setup (constraints + indexes)
    │   └── graph_ingestor.py      # batched Neo4j ingestion, checkpoint resume
    ├── vector/
    │   ├── chroma_client.py       # ChromaDB client, cosine collection
    │   └── vector_ingestor.py     # batch embedding + upsert, checkpoint resume
    ├── retrieval/
    │   ├── graph_retriever.py     # Cypher entity + topic matching, temporal filter
    │   └── vector_retriever.py    # semantic search, metadata pre-filtering
    ├── fusion/
    │   └── rrf.py                 # Reciprocal Rank Fusion, hybrid source tagging
    └── generation/
        └── answer_generator.py    # Claude Sonnet, cited answers, temporal prompting
```

---

## Data Stats

| Metric | Value |
|---|---|
| Raw items collected | 11,536 |
| Items successfully extracted | 11,048 |
| Extraction success rate | 95.8% |
| Neo4j — Post nodes | 1,215 |
| Neo4j — Comment nodes | 9,833 |
| Neo4j — Author nodes | 4,750 (deduplicated) |
| Neo4j — Entity nodes | 5,953 (deduplicated) |
| Neo4j — Topic nodes | 727 (deduplicated) |
| Neo4j — Subreddit nodes | 3 |
| Neo4j — Total relationships | 54,588 |
| ChromaDB — Vectors | 11,048 |
| ChromaDB — Embedding dimensions | 384 |

---

## Module Status

| Module | Status | Description |
|---|---|---|
| `reddit_scraper.py` | ✅ Complete | Arctic Shift scraper, 5 subs × 3 windows, 11,536 items |
| `extractor.py` | ✅ Complete | Groq extraction, 6-key auto-rotation, Gemini fallback, checkpoint resume |
| `neo4j_client.py` | ✅ Complete | Schema setup — 6 constraints, 7 indexes |
| `graph_ingestor.py` | ✅ Complete | 11,048 nodes ingested, 54,588 relationships, batched + resumable |
| `chroma_client.py` | ✅ Complete | Persistent ChromaDB collection, cosine similarity |
| `vector_ingestor.py` | ✅ Complete | 11,048 vectors, batched embedding on Apple MPS, checkpoint resume |
| `graph_retriever.py` | ✅ Complete | Entity match + topic match Cypher, temporal window filtering |
| `vector_retriever.py` | ✅ Complete | Semantic search, metadata pre-filtering, comparison retrieval |
| `rrf.py` | ✅ Complete | Reciprocal Rank Fusion, hybrid source tagging, variadic list support |
| `answer_generator.py` | ✅ Complete | Claude Sonnet, inline citations, temporal-aware prompting, query type hints |
| `demo.py` | ✅ Complete | 4-query end-to-end: semantic, graph, hybrid, time-series |
