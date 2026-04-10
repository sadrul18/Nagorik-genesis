"""
Web Knowledge Retrieval for NAGORIK-GENESIS.
Search the web for real-world context about a policy, cache results,
and inject as knowledge context into LLM prompts.

Supports two backends:
  - Tavily  (primary, RAG-optimized, needs API key)
  - DuckDuckGo (free fallback, no API key needed)

Usage:
    client = WebKnowledgeClient(tavily_api_key="tvly-...", ollama_host="http://localhost:11434")
    context = client.search_policy_context("Free Healthcare", "Universal healthcare...", "Healthcare")
"""
import hashlib
import json
import logging
import os
import re
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Cache TTL in hours (7 days)
CACHE_TTL_HOURS = 168

# Max search queries per policy
MAX_QUERIES = 5

# Max raw snippets to collect before summarization
MAX_SNIPPETS = 15


class WebKnowledgeClient:
    """Search the web for real-world context about a policy, cache results."""

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        cache_dir: str = "knowledge_cache",
        ollama_model: str = "qwen2.5:7b",
        ollama_host: str = "http://localhost:11434",
    ):
        self.tavily_api_key = tavily_api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host.rstrip("/")

        # Lazy-initialize Tavily client
        self._tavily_client = None
        if self.tavily_api_key:
            try:
                from tavily import TavilyClient
                self._tavily_client = TavilyClient(api_key=self.tavily_api_key)
                logger.info("Tavily client initialized (primary search backend)")
            except ImportError:
                logger.warning("tavily-python not installed — falling back to DuckDuckGo")
            except Exception as e:
                logger.warning(f"Tavily init failed: {e} — falling back to DuckDuckGo")

        # Check DuckDuckGo availability
        self._ddg_available = False
        try:
            from ddgs import DDGS  # noqa: F401
            self._ddg_available = True
            logger.info("DuckDuckGo fallback available")
        except ImportError:
            try:
                from duckduckgo_search import DDGS  # noqa: F401
                self._ddg_available = True
                logger.info("DuckDuckGo fallback available (legacy package)")
            except ImportError:
                logger.warning("duckduckgo-search not installed — web knowledge disabled")

    # ── Public API ──────────────────────────────────────────────────────────

    def search_policy_context(
        self,
        policy_title: str,
        policy_description: str,
        policy_domain: str,
    ) -> str:
        """
        Main entry point. Returns a knowledge context string for prompt injection.

        Steps:
        1. Check cache (by policy hash)
        2. If miss → generate search queries
        3. Search via Tavily (or DuckDuckGo fallback)
        4. Summarize raw results via Ollama
        5. Cache result
        6. Return context string

        Returns empty string if all backends fail (graceful degradation).
        """
        cache_key = self._get_cache_key(policy_title, policy_domain)

        # Step 1: Check cache
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info(f"Web knowledge cache HIT for '{policy_title}' [{cache_key}]")
            return cached["context_summary"]

        logger.info(f"Web knowledge cache MISS for '{policy_title}' — searching web...")

        # Step 2: Generate search queries
        queries = self._generate_search_queries(policy_title, policy_description, policy_domain)
        if not queries:
            logger.warning("No search queries generated — skipping web knowledge")
            return ""

        # Step 3: Search web
        raw_results = self._search_web(queries)
        if not raw_results:
            logger.warning("No web results found — skipping web knowledge")
            return ""

        # Step 4: Summarize with Ollama
        context_summary = self._summarize_with_ollama(raw_results, policy_title, policy_domain)
        if not context_summary:
            # Fallback: use raw snippets directly (truncated)
            context_summary = self._format_raw_snippets(raw_results)

        # Step 5: Cache
        sources = list({r["url"] for r in raw_results if r.get("url")})
        self._save_cache(cache_key, {
            "policy_title": policy_title,
            "policy_domain": policy_domain,
            "queries_used": queries,
            "raw_results_count": len(raw_results),
            "context_summary": context_summary,
            "sources": sources[:10],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ttl_hours": CACHE_TTL_HOURS,
        })

        logger.info(
            f"Web knowledge cached for '{policy_title}' "
            f"[{len(raw_results)} results, {len(sources)} sources]"
        )
        return context_summary

    def is_available(self) -> bool:
        """Check if any search backend is available."""
        return self._tavily_client is not None or self._ddg_available

    # ── Search Query Generation ─────────────────────────────────────────────

    def _generate_search_queries(
        self,
        title: str,
        description: str,
        domain: str,
    ) -> List[str]:
        """Generate targeted search queries. Uses Ollama (Tier 2), falls back to templates (Tier 1)."""
        # Try Tier 2: LLM-generated queries via Ollama
        try:
            queries = self._generate_queries_ollama(title, description, domain)
            if queries and len(queries) >= 3:
                logger.info(f"Generated {len(queries)} search queries via Ollama")
                return queries[:MAX_QUERIES]
        except Exception as e:
            logger.warning(f"Ollama query generation failed: {e}")

        # Tier 1 fallback: template-based queries
        logger.info("Using template-based search queries (Tier 1 fallback)")
        return self._generate_queries_template(title, domain)

    def _generate_queries_ollama(self, title: str, description: str, domain: str) -> List[str]:
        """Use local Ollama to generate search queries."""
        system = (
            "You generate web search queries to find real-world evidence about a policy. "
            "Focus on Bangladesh and South Asia. Output ONLY a JSON array of 5 strings."
        )
        user = (
            f"Policy for Bangladesh:\n"
            f"Title: {title}\n"
            f"Domain: {domain}\n"
            f"Description: {description}\n\n"
            f"Generate 5 web search queries to find:\n"
            f"1. Similar policies implemented in Bangladesh\n"
            f"2. Economic impact data from Bangladesh or South Asia\n"
            f"3. News articles about this topic in Bangladesh\n"
            f"4. Case studies from comparable developing countries\n"
            f"5. Expert analysis or policy reports\n\n"
            f"Return ONLY a JSON array of 5 search query strings."
        )
        response = self._call_ollama(system, user, force_json=True)
        if not response:
            return []

        # Parse JSON array from response
        try:
            # Try direct parse
            queries = json.loads(response)
            if isinstance(queries, list):
                return [str(q) for q in queries if isinstance(q, str)]
        except json.JSONDecodeError:
            pass

        # Try extracting array from response
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            try:
                queries = json.loads(match.group(0))
                if isinstance(queries, list):
                    return [str(q) for q in queries if isinstance(q, str)]
            except json.JSONDecodeError:
                pass

        return []

    def _generate_queries_template(self, title: str, domain: str) -> List[str]:
        """Template-based fallback for query generation (no LLM needed)."""
        return [
            f"Bangladesh {domain} {title} impact",
            f"{title} policy developing countries results",
            f"Bangladesh {domain} budget allocation outcomes",
            f"{title} South Asia case study economic impact",
            f"Bangladesh {title} news 2024 2025 2026",
        ]

    # ── Web Search Backends ─────────────────────────────────────────────────

    def _search_web(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Search using Tavily (primary) or DuckDuckGo (fallback)."""
        # Try Tavily first
        if self._tavily_client:
            try:
                results = self._search_tavily(queries)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}")

        # Fallback to DuckDuckGo
        if self._ddg_available:
            try:
                results = self._search_ddg(queries)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"DuckDuckGo search failed: {e}")

        return []

    def _search_tavily(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Search using Tavily API — returns deduplicated results."""
        all_results = []
        seen_urls = set()

        for query in queries:
            try:
                response = self._tavily_client.search(
                    query=query,
                    search_depth="basic",
                    max_results=5,
                    include_answer=False,
                )
                for r in response.get("results", []):
                    url = r.get("url", "")
                    if url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append({
                            "title": r.get("title", ""),
                            "snippet": r.get("content", ""),
                            "url": url,
                            "source": "tavily",
                        })
            except Exception as e:
                logger.warning(f"Tavily query '{query[:50]}...' failed: {e}")
                continue

            if len(all_results) >= MAX_SNIPPETS:
                break

        return all_results[:MAX_SNIPPETS]

    def _search_ddg(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo — free, no API key."""
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        all_results = []
        seen_urls = set()

        ddgs = DDGS(timeout=15)

        for query in queries:
            try:
                # Text search
                text_results = ddgs.text(query, max_results=4)
                for r in (text_results or []):
                    url = r.get("href", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append({
                            "title": r.get("title", ""),
                            "snippet": r.get("body", ""),
                            "url": url,
                            "source": "duckduckgo",
                        })
            except Exception as e:
                logger.warning(f"DDG text query '{query[:50]}...' failed: {e}")

            try:
                # News search for recency
                news_results = ddgs.news(query, max_results=2)
                for r in (news_results or []):
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append({
                            "title": r.get("title", ""),
                            "snippet": r.get("body", ""),
                            "url": url,
                            "source": "duckduckgo_news",
                        })
            except Exception as e:
                logger.warning(f"DDG news query '{query[:50]}...' failed: {e}")

            if len(all_results) >= MAX_SNIPPETS:
                break

        return all_results[:MAX_SNIPPETS]

    # ── Summarization ───────────────────────────────────────────────────────

    def _summarize_with_ollama(
        self,
        raw_results: List[Dict[str, Any]],
        policy_title: str,
        policy_domain: str,
    ) -> str:
        """Use local Ollama to distill raw search results into a knowledge block."""
        # Build snippets text
        snippets_text = ""
        for i, r in enumerate(raw_results, 1):
            snippets_text += f"\n[{i}] {r.get('title', 'No title')}\n"
            snippet = r.get("snippet", "")
            if snippet:
                snippets_text += f"    {snippet[:500]}\n"
            snippets_text += f"    Source: {r.get('url', 'N/A')}\n"

        system = (
            "You are a Bangladesh policy research analyst. You distill web search results "
            "into concise, fact-based bullet points relevant to a specific policy. "
            "Focus on: concrete numbers, dates, outcomes, and Bangladesh-specific context. "
            "Output 8-15 bullet points. Each bullet should be one factual sentence. "
            "Do NOT make up facts — only summarize what the search results contain. "
            "If results are irrelevant, say so honestly. Output plain text bullets starting with '- '."
        )

        user = (
            f"Policy: {policy_title} (Domain: {policy_domain})\n\n"
            f"Web search results to summarize:\n{snippets_text}\n\n"
            f"Summarize into 8-15 bullet points relevant to this policy in Bangladesh context. "
            f"Include specific data points (numbers, dates, BDT amounts, percentages) where available. "
            f"Start each bullet with '- '."
        )

        response = self._call_ollama(system, user, force_json=False)
        if not response:
            return ""

        # Extract bullet points — filter for lines starting with '-' or '•'
        lines = response.strip().split("\n")
        bullets = []
        for line in lines:
            line = line.strip()
            if line.startswith(("- ", "• ", "* ")):
                bullets.append(line)
            elif line.startswith(("-", "•", "*")) and len(line) > 2:
                bullets.append("- " + line[1:].strip())

        if not bullets:
            # If Ollama didn't format as bullets, use the raw response
            return response.strip()

        return "\n".join(bullets[:15])

    def _format_raw_snippets(self, raw_results: List[Dict[str, Any]]) -> str:
        """Fallback: format raw snippets directly (when Ollama summarization fails)."""
        bullets = []
        for r in raw_results[:10]:
            snippet = r.get("snippet", "").strip()
            if snippet:
                # Truncate long snippets
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                bullets.append(f"- {snippet}")
        return "\n".join(bullets)

    # ── Ollama Helper ───────────────────────────────────────────────────────

    def _call_ollama(self, system: str, user: str, force_json: bool = False) -> str:
        """Send a chat request to Ollama and return response text."""
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        if force_json:
            payload["format"] = "json"

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.ollama_host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode())
                return body["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return ""

    # ── Cache Management ────────────────────────────────────────────────────

    def _get_cache_key(self, title: str, domain: str) -> str:
        """SHA256 hash of policy title + domain for cache lookup."""
        raw = f"{title.strip().lower()}|{domain.strip().lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def _load_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached knowledge if exists and not expired."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check TTL
            created_at = datetime.fromisoformat(data["created_at"])
            ttl_hours = data.get("ttl_hours", CACHE_TTL_HOURS)
            age_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600

            if age_hours > ttl_hours:
                logger.info(f"Cache expired for {cache_key} (age={age_hours:.1f}h > ttl={ttl_hours}h)")
                cache_file.unlink()
                return None

            return data
        except Exception as e:
            logger.warning(f"Cache read error for {cache_key}: {e}")
            return None

    def _save_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save knowledge context to cache file."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Cache write error for {cache_key}: {e}")


def create_web_knowledge_client(
    tavily_api_key: Optional[str] = None,
    cache_dir: str = "knowledge_cache",
    ollama_model: str = "qwen2.5:7b",
    ollama_host: str = "http://localhost:11434",
) -> WebKnowledgeClient:
    """Factory function matching the project's pattern."""
    return WebKnowledgeClient(
        tavily_api_key=tavily_api_key,
        cache_dir=cache_dir,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
    )
