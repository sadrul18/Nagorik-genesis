# Web Knowledge Base Integration Plan

Enrich LLM prompts with real-world context scraped from the web so that policy simulations reflect actual incidents, data, and precedents — not just the static Bangladesh context hardcoded in the system prompt.

---

## 1. Problem Statement

The current system prompt contains a **fixed Bangladesh context block** (~25 lines) covering generic facts — GDP growth, RMG sector, population density, etc. When a user simulates a novel policy like *"Tax on freelance digital income"* or *"Flood early-warning system via mobile"*, the LLM has no policy-specific real-world grounding and must hallucinate the socioeconomic impact.

**What we need:** Before simulation starts, search the web for similar policies/events/incidents in Bangladesh (or comparable countries), extract key facts, and inject them into the LLM prompt as a **knowledge context block**.

---

## 2. Web Search API Comparison

| Criterion | DuckDuckGo (`ddgs`) | Tavily | SerpAPI | Brave Search API | Google Custom Search |
|---|---|---|---|---|---|
| **Free tier** | Unlimited (no API key) | 1,000 credits/mo | 250 searches/mo | ~1,000 searches/mo ($5 credit) | 100 queries/day (~3,000/mo) |
| **Paid entry** | Free forever | $30/mo (4,000 credits) | $25/mo (1,000) | $5/1,000 requests | $5/1,000 queries |
| **API key required** | No | Yes | Yes | Yes | Yes + Custom Search Engine ID |
| **Python package** | `pip install ddgs` | `pip install tavily-python` | `pip install google-search-results` | `requests` (REST only) | `pip install google-api-python-client` |
| **RAG-optimized** | No (raw snippets) | **Yes** (`get_search_context()`) | No (raw SERP data) | Partial (LLM context endpoint) | No |
| **News search** | Yes (`ddgs.news()`) | Yes (topic filter) | Yes (`tbm=nws`) | Yes (news endpoint) | No built-in |
| **Content extraction** | No | **Yes** (`extract()` from URLs) | No | No | No |
| **Rate limit risk** | High (IP-based, can get blocked) | Low (token-based) | Low (token-based) | Low (token-based) | Low (token-based) |
| **Result quality** | Moderate (Bing-backed) | **High** (AI-curated) | **High** (actual Google SERP) | High (own index, 30B pages) | **High** (Google) |
| **Bangladesh/South Asia coverage** | Good | Good | **Best** (Google index) | Good | **Best** (Google index) |
| **Setup complexity** | Trivial (0 config) | Easy (1 env var) | Easy (1 env var) | Easy (1 header) | Complex (GCP project + CSE) |
| **Offline/local dev** | Works without key | Needs key | Needs key | Needs key | Needs key |

### Verdict

| Rank | Tool | Role | Why |
|---|---|---|---|
| **1st** | **Tavily** | Primary search engine | Built for RAG. `get_search_context()` returns LLM-ready text. 1,000 free credits/month is enough for development + moderate production use (~14 simulations/day × 72 anchor searches = too many → we cache per policy, so 1 search per unique policy). |
| **2nd** | **DuckDuckGo (`ddgs`)** | Free fallback | Zero cost, zero config. Use when Tavily quota is exhausted or API key isn't set. Quality is lower but better than nothing. |
| **Skip** | SerpAPI | — | Expensive for what it offers. Only 250 free searches. No RAG features. |
| **Skip** | Brave | — | Good quality but REST-only (no Python SDK), and $5/mo minimum for meaningful use. |
| **Skip** | Google CSE | — | Requires GCP project setup, Custom Search Engine creation — too much overhead for this project. |

**Recommendation: Tavily (primary) + DuckDuckGo (free fallback)**

---

## 3. Architecture

### 3.1 Data Flow

```
User enters policy → [web_knowledge.py] → Search web → Extract + Summarize → Cache → Inject into LLM prompt
                          │                                                       │
                          ├── Tavily search (primary)                             ├── knowledge_cache/
                          └── DuckDuckGo (fallback)                               │   ├── <policy_hash>.json
                                                                                  │   └── ...
                                                                                  │
                                                                                  └── Injected as REAL-WORLD CONTEXT block
                                                                                      in system prompt
```

### 3.2 New Module: `web_knowledge.py`

```python
# web_knowledge.py — Web knowledge retrieval for policy context enrichment

class WebKnowledgeClient:
    """Search the web for real-world context about a policy, cache results."""

    def __init__(self, tavily_api_key=None, cache_dir="knowledge_cache"):
        ...

    def search_policy_context(self, policy_title, policy_description, policy_domain) -> str:
        """
        Main entry point. Returns a knowledge context string ready for prompt injection.

        Steps:
        1. Check cache (by policy hash)
        2. If miss → generate search queries from policy
        3. Search via Tavily (or DuckDuckGo fallback)
        4. Deduplicate + summarize via Ollama
        5. Cache result
        6. Return context string
        """
        ...

    def _generate_search_queries(self, title, description, domain) -> list[str]:
        """Generate 3-5 targeted search queries from the policy.
        Example policy: "Free universal healthcare"
        Queries:
          - "Bangladesh universal healthcare policy impact"
          - "free healthcare developing countries economic impact"
          - "Bangladesh healthcare budget allocation results"
          - "universal healthcare South Asia case study"
        """
        ...

    def _search_tavily(self, queries) -> list[dict]:
        """Search using Tavily API with get_search_context()."""
        ...

    def _search_ddg(self, queries) -> list[dict]:
        """Fallback: search using DuckDuckGo."""
        ...

    def _summarize_with_ollama(self, raw_results, policy) -> str:
        """Use local Ollama to distill raw search results into a concise
        knowledge block (10-15 bullet points, Bangladesh-relevant)."""
        ...

    def _get_cache_key(self, title, description, domain) -> str:
        """SHA256 hash of policy for cache lookup."""
        ...
```

### 3.3 Prompt Injection Point

Current system prompt structure:
```
CRITICAL CONTEXT — BANGLADESH (বাংলাদেশ):
- Currency: BDT...
- Income context...
- Economy...
[... static block ...]
```

New structure:
```
CRITICAL CONTEXT — BANGLADESH (বাংলাদেশ):
- Currency: BDT...
[... same static block ...]

REAL-WORLD EVIDENCE FOR THIS POLICY:
The following facts were retrieved from web sources about similar policies/events:
- [bullet 1 from web search]
- [bullet 2 from web search]
- ...
- [bullet N from web search]
Sources: [url1], [url2], ...

Use these real-world precedents to ground your reaction prediction.
```

### 3.4 Cache Strategy

```
knowledge_cache/
├── a3f8b2c1d4e5.json       # SHA256(policy_title + domain)[:12]
│   {
│     "policy_title": "Free universal healthcare",
│     "policy_domain": "Healthcare",
│     "queries_used": ["Bangladesh universal healthcare...", ...],
│     "raw_results_count": 15,
│     "context_summary": "- In 2023, Bangladesh allocated 5.4% of budget...\n- ...",
│     "sources": ["https://...", "https://..."],
│     "created_at": "2026-04-10T12:00:00",
│     "ttl_hours": 168
│   }
```

- **TTL: 7 days** — web facts about policy precedents don't change hourly
- **One search per unique policy** — not per citizen, not per step
- **Cache hit = zero API calls** — simulation can run unlimited times on cached knowledge

### 3.5 Search Query Generation Strategy

The quality of the knowledge base depends entirely on query formulation. Use a 2-tier approach:

**Tier 1: Template-based queries (fast, no LLM needed)**
```python
QUERY_TEMPLATES = [
    "Bangladesh {domain} {title} impact",
    "{title} policy developing countries results",
    "Bangladesh {domain} budget allocation outcomes",
    "{title} South Asia case study economic impact",
    "Bangladesh {title} news 2024 2025",
]
```

**Tier 2: LLM-generated queries (better, uses Ollama)**
```python
prompt = f"""Given this policy for Bangladesh:
Title: {title}
Domain: {domain}
Description: {description}

Generate 5 web search queries to find real-world precedents,
similar policies in other developing countries, and economic
impact data. Focus on Bangladesh and South Asia.
Return as JSON array of strings."""
```

**Default: Tier 2 (LLM-generated)** — uses local Ollama, zero cost, better query diversity.
**Fallback: Tier 1** — if Ollama is temporarily unavailable during query generation.

---

## 4. Integration Points

### 4.1 Files to Modify

| File | Change |
|---|---|
| `web_knowledge.py` | **New file** — `WebKnowledgeClient` class |
| `llm_client.py` | Add `knowledge_context` parameter to `generate_citizen_reaction()`. Inject into system prompt. |
| `simulation.py` | Call `web_knowledge.search_policy_context()` once at simulation start, pass context string through to LLM calls. |
| `app.py` | Initialize `WebKnowledgeClient`, show "Searching web for policy context..." status in UI. |
| `config.py` | Add `tavily_api_key` to `Settings`. |
| `requirements.txt` | Add `tavily-python`, `ddgs`. |

### 4.2 Simulation Flow (Updated)

```
1. User clicks "Run Simulation"
2. [NEW] web_knowledge.search_policy_context(policy) → knowledge_context string
   - Cache hit → instant (<1ms)
   - Cache miss → Tavily search (2-3s) → Ollama summarize (5-10s) → cache
3. run_simulation() receives knowledge_context
4. For each LLM call (anchor citizens):
   - System prompt now includes knowledge_context block
   - LLM grounds its reaction on real-world evidence
5. NN predictions for remaining citizens are calibrated against LLM anchors
   (anchors are already informed by web knowledge → calibration propagates)
```

### 4.3 Cost Analysis

| Scenario | Tavily Credits Used | DuckDuckGo Calls |
|---|---|---|
| First run of a new policy | 5 searches × 1 credit = **5 credits** | 0 |
| Repeat run of same policy | **0** (cached) | 0 |
| 200 unique policies/month | **1,000 credits** (fits free tier) | 0 |
| Tavily key not set | 0 | 5 searches (free) |
| Tavily quota exhausted | 0 | 5 searches (fallback) |

**For typical usage (10-20 unique policies/month), the free Tavily tier is more than sufficient.**

---

## 5. Implementation Plan

### Phase 1: Core Module (1 file, testable independently)
- [ ] Create `web_knowledge.py` with `WebKnowledgeClient`
- [ ] Implement Tavily search with `get_search_context()`
- [ ] Implement DuckDuckGo fallback with `DDGS().text()` + `DDGS().news()`
- [ ] Implement file-based caching with TTL
- [ ] Implement Ollama-based summarization of raw results
- [ ] Implement LLM-based query generation (Tier 2)
- [ ] Add template fallback query generation (Tier 1)

### Phase 2: Integration
- [ ] Add `tavily_api_key` to `config.py` Settings
- [ ] Update `llm_client.py` — add `knowledge_context` param to both `OllamaClient` and `GeminiClient`
- [ ] Update `simulation.py` — call `search_policy_context()` once before step loop
- [ ] Update `app.py` — initialize client, show search status in UI
- [ ] Update `requirements.txt`

### Phase 3: Testing & Validation
- [ ] Test with policy: "Free universal healthcare" (cache miss → Tavily → summarize → cache)
- [ ] Test cache hit (rerun same policy)
- [ ] Test DuckDuckGo fallback (no Tavily key)
- [ ] Compare LLM reactions with vs without web knowledge
- [ ] Run HYBRID simulation end-to-end with web knowledge enabled

---

## 6. Example: Before vs After

### Policy: "Tax on freelance digital income in Bangladesh"

**Before (no web knowledge):**
The LLM only knows generic Bangladesh facts. It guesses that freelancers might be unhappy. No grounding in actual policy debates or income data.

**After (with web knowledge):**
```
REAL-WORLD EVIDENCE FOR THIS POLICY:
- Bangladesh earned $800M+ from IT freelancing in 2024, with 650,000+ registered freelancers (BIDA data)
- In 2023, NBR proposed 10% withholding tax on freelance income received via mobile banking; freelancer associations protested
- India's 30% crypto/digital income tax (2022) caused significant capital flight to Dubai; Bangladesh freelancers cited similar concerns
- Philippines exempts freelancers earning below ₱250,000/year; graduated tax above that threshold
- bKash/Nagad processed ৳12.5 trillion in mobile transactions in 2024; digital income tracking is technically feasible
- BASIS (Bangladesh Association of Software and Information Services) lobbied for 5-year tax holiday for IT sector
Sources: bdnews24.com, thedailystar.net, lightcastlebd.com, bida.gov.bd
```

Now the LLM can predict: a low-income freelancer earning ৳15,000/month will react very differently from a high-income tech professional earning ৳200,000/month, grounded in actual tax threshold debates.

---

## 7. Fallback Behavior

```
Priority chain:
1. Cache hit → use cached knowledge (instant)
2. Tavily API → search + extract + summarize (5-15s)
3. DuckDuckGo → search + summarize (3-10s)
4. No internet / all fail → run without web knowledge (current behavior, no regression)
```

The system **never blocks** on web search failure. Worst case = current behavior (static context only).

---

## 8. Security & Privacy Considerations

- Tavily API key stored in `.env` file (already gitignored via `dotenv` usage in `config.py`)
- DuckDuckGo requires no credentials
- Search queries contain only policy text (no citizen PII)
- Cached results stored locally in `knowledge_cache/` (add to `.gitignore`)
- Ollama summarization runs entirely locally — no data leaves the machine except search queries
- Web content is summarized, not stored raw — reduces risk of injecting malicious content into prompts
