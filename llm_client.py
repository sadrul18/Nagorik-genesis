"""LLM client for নাগরিক-GENESIS.
Supports two backends:
  - ollama  : fully local, unlimited, no quota (default)
  - gemini  : Google Gemini API with automatic key rotation
4 expert perspectives: Economist, Activist, Garment Industry, Rural Leader.
"""
import json
import re
import logging
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List
from google import genai
from google.genai import errors as genai_errors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.5-flash"


class GeminiClient:
    """Client for interacting with Google's Gemini API with automatic key rotation."""

    def __init__(self, api_key: str, backup_keys: Optional[List[str]] = None):
        """
        Initialize the Gemini client with API key rotation support.

        Args:
            api_key: Primary Google API key for Gemini.
            backup_keys: Optional list of backup API keys to rotate through.
        """
        self.api_keys = [api_key]
        if backup_keys:
            self.api_keys.extend(backup_keys)

        self.current_key_index = 0
        self.exhausted_keys = set()
        self.all_keys_exhausted = False

        self._switch_api_key(0)

        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 4.5  # 60s / 15 requests = 4s + buffer

    def _switch_api_key(self, key_index: int) -> bool:
        """Switch to a different API key."""
        if key_index >= len(self.api_keys):
            logger.error("No more API keys available!")
            return False

        self.current_key_index = key_index
        current_key = self.api_keys[key_index]

        self.client = genai.Client(api_key=current_key)

        masked_key = f"{current_key[:8]}...{current_key[-4:]}" if len(current_key) > 12 else "***"
        logger.info(f"🔄 Switched to API key #{key_index + 1}/{len(self.api_keys)} ({masked_key})")
        return True

    def _rotate_to_next_key(self) -> bool:
        """Rotate to the next available API key."""
        self.exhausted_keys.add(self.current_key_index)

        for i in range(len(self.api_keys)):
            if i not in self.exhausted_keys:
                if self._switch_api_key(i):
                    logger.info(f"✅ Successfully rotated to backup key #{i + 1}")
                    return True

        self.all_keys_exhausted = True
        logger.error("❌ All API keys have exhausted their daily quota!")
        return False

    def _rate_limit(self):
        """Enforce rate limiting to stay within free tier quota (15 req/min)."""
        self.request_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.info(f"Rate limiting: sleeping {sleep_time:.2f}s (request #{self.request_count})")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _call_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        Call Gemini API with exponential backoff retry logic and automatic key rotation.

        Args:
            prompt: The prompt to send to the model.
            max_retries: Maximum number of retry attempts per key.

        Returns:
            Response text or None if all retries fail.
        """
        if self.all_keys_exhausted:
            return None

        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = self.client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower() or "RESOURCE_EXHAUSTED" in error_msg:
                    if "PerDay" in error_msg or "limit: 0" in error_msg:
                        logger.warning("⚠️ Daily quota exceeded for current API key")

                        if self._rotate_to_next_key():
                            logger.info("🔄 Continuing with backup key...")
                            continue
                        else:
                            logger.error("❌ All API keys exhausted. Cannot continue.")
                            return None

                    retry_match = re.search(r'retry in ([\d.]+)s', error_msg)
                    if retry_match:
                        retry_delay = float(retry_match.group(1)) + 1
                    else:
                        retry_delay = (2 ** attempt) * 5

                    if retry_delay < 120 and attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit, retrying in {retry_delay:.1f}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Rate limit exceeded with long delay ({retry_delay:.1f}s) or max retries reached")
                        return None
                else:
                    logger.error(f"Error calling Gemini API: {e}")
                    return None

        return None

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response text."""
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                json_str = match.group(0)
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to extract JSON from response: {text[:200]}")
            return None

    def generate_citizen_reaction(
        self,
        citizen_profile: Dict[str, Any],
        current_state: Dict[str, Any],
        policy: Dict[str, Any],
        knowledge_context: str = ""
    ) -> Dict[str, Any]:
        """
        Generate a Bangladeshi citizen's reaction to a policy using Gemini.

        Args:
            citizen_profile: Dict with citizen attributes (age, income_level, profession, division, religion, etc.).
            current_state: Dict with current happiness, policy_support, income.
            policy: Dict with title, description, domain.
            knowledge_context: Optional web-sourced knowledge to ground the response.

        Returns:
            Dict with new_happiness, new_policy_support, income_delta, short_reason, diary_entry.
        """
        system_prompt = """You are simulating how a fictional Bangladeshi citizen (নাগরিক) reacts to a new government policy or event.
You receive citizen_profile, current_state, and policy.

CRITICAL CONTEXT — BANGLADESH (বাংলাদেশ):
- Currency: BDT (৳ Bangladeshi Taka). All income values are MONTHLY in BDT. ৳1 USD ≈ ৳110 BDT.
- Income context: ৳8,000-15,000/month = low income (গরিব). ৳20,000-60,000 = middle class (মধ্যবিত্ত). ৳80,000+ = upper class (উচ্চবিত্ত).
- Economy: 7th largest by PPP in Asia. GDP growth ~6-7%. Inflation ~9%.
- Key sectors: Ready-Made Garments (RMG) employs 4M+ workers (80% women). Agriculture employs 40% of workforce. Remittance is 6% of GDP.
- Geography: Bangladesh is a low-lying delta, extremely vulnerable to floods, cyclones, and rising sea levels. Northern districts face drought.
- Social structure: Joint families common. 91% Muslim majority. Strong community bonds. Hierarchical respect structures.
- Urban reality: Dhaka is among the most densely populated cities (47,000 people/km²). bosti (slums) house ~40% of urban population.
- Rural reality: 63% of population is rural. Agriculture, fishing, and remittance are primary income sources.
- Education: Madrasa system runs parallel to secular education. Female education has improved dramatically but gender gaps persist in workforce.
- Digital: 130M+ internet users. bKash/Nagad mobile financial services widely used. Growing IT freelancing sector.

CITIZEN ATTRIBUTE GUIDANCE:
- Respect profession context: A garment worker (গার্মেন্ট শ্রমিক) earning ৳12,000/month will react very differently to a wage policy than a corporate executive earning ৳300,000.
- Respect city_zone: A bosti resident faces existential risk from eviction policies. A graam (village) resident cares deeply about crop prices and flood relief.
- Respect political_view: government_supporter vs opposition_supporter will interpret the same policy differently.
- Respect is_remittance_family: Families receiving remittance have different financial resilience and priorities.
- Respect division: Dhaka/Chittagong urban dwellers vs Rangpur/Barisal rural dwellers have different concerns.

Output ONLY valid JSON with these exact keys:
- new_happiness: float between 0 and 1
- new_policy_support: float between -1 and 1
- income_delta: float in BDT (change in MONTHLY income, can be negative, zero, or positive)
- short_reason: string, 1-2 sentences explaining the reaction
- diary_entry: string, 3-5 sentences in first-person perspective as a Bangladeshi citizen. May include Bengali words/phrases naturally.

Do not include any explanation outside the JSON."""

        if knowledge_context:
            knowledge_block = (
                "\n\nREAL-WORLD CONTEXT (from web search — use to ground your response):\n"
                + knowledge_context
                + "\n"
            )
            marker = "Output ONLY valid JSON"
            if marker in system_prompt:
                system_prompt = system_prompt.replace(marker, knowledge_block + "\n" + marker)
            else:
                system_prompt = system_prompt + knowledge_block

        user_prompt = f"""Citizen Profile:
{json.dumps(citizen_profile, indent=2)}

Current State:
{json.dumps(current_state, indent=2)}

Policy:
{json.dumps(policy, indent=2)}

Generate the citizen's reaction as JSON."""

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        response_text = self._call_with_retry(full_prompt)
        if not response_text:
            raise RuntimeError("LLM call failed: no response after retries (quota exhausted or API error)")

        result_json = self._extract_json(response_text)
        if not result_json:
            raise RuntimeError(f"LLM call failed: could not parse JSON from response: {response_text[:200]}")

        result_json["new_happiness"] = max(0.0, min(1.0, float(result_json.get("new_happiness", 0.5))))
        result_json["new_policy_support"] = max(-1.0, min(1.0, float(result_json.get("new_policy_support", 0.0))))
        result_json["income_delta"] = float(result_json.get("income_delta", 0.0))
        result_json["short_reason"] = str(result_json.get("short_reason", "No reason provided."))
        result_json["diary_entry"] = str(result_json.get("diary_entry", "No diary entry."))
        return result_json

    def generate_expert_summary(
        self,
        step_stats: Dict[str, Any],
        policy: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate 4 Bangladeshi expert perspectives on simulation results.

        Args:
            step_stats: Aggregated statistics for the current step.
            policy: Policy information.

        Returns:
            Dict with economist_view, activist_view, garment_industry_view, rural_leader_view.
        """
        system_prompt = """You are analyzing a synthetic Bangladesh society simulation. You have aggregated metrics from a policy simulation affecting Bangladeshi citizens across different income levels, city zones, and divisions.

Generate FOUR expert perspectives grounded in Bangladeshi socioeconomic reality:

1. অর্থনীতিবিদ (Economist): Focus on BDT income effects, inflation impact, GDP implications, remittance effects, employment shifts, and fiscal sustainability. Reference Bangladesh Bank data context where relevant.

2. সমাজকর্মী (Social Activist / NGO Leader): Focus on poverty impact, bosti (slum) dwellers, garment workers' welfare, gender equity, child labor risk, and marginalized communities (Hijra, ethnic minorities). Reference NGO sector context (BRAC, Grameen).

3. গার্মেন্ট শিল্প প্রতিনিধি (Garment Industry Representative): Focus on RMG sector competitiveness, worker retention, export impact, buyer compliance requirements, factory-level effects. This sector is 84% of Bangladesh's $40B+ exports.

4. গ্রামীণ নেতা (Rural Community Leader): Focus on agricultural impact, rural employment, climate vulnerability, flood/cyclone resilience, food security, and rural-urban migration pressure. 63% of Bangladesh is rural.

Use the simulation metrics to ground your reasoning. Highlight which demographic groups in Bangladesh are most affected.

Output ONLY valid JSON with these exact keys:
- economist_view: string, 3-5 sentences
- activist_view: string, 3-5 sentences
- garment_industry_view: string, 3-5 sentences
- rural_leader_view: string, 3-5 sentences

Do not include any explanation outside the JSON."""

        user_prompt = f"""Policy:
{json.dumps(policy, indent=2)}

Current Step Statistics:
{json.dumps(step_stats, indent=2)}

Generate expert perspectives as JSON."""

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            response_text = self._call_with_retry(full_prompt)
            if not response_text:
                raise Exception("Failed to get response after retries")

            result_json = self._extract_json(response_text)

            if result_json:
                return {
                    "economist_view": str(result_json.get("economist_view", "No economist view available.")),
                    "activist_view": str(result_json.get("activist_view", "No activist view available.")),
                    "garment_industry_view": str(result_json.get("garment_industry_view", "No garment industry view available.")),
                    "rural_leader_view": str(result_json.get("rural_leader_view", "No rural leader view available."))
                }
            else:
                logger.warning("Failed to parse expert summary, using fallback")
                return {
                    "economist_view": "বিশ্লেষণ পার্সিং ত্রুটির কারণে অনুপলব্ধ।",
                    "activist_view": "বিশ্লেষণ পার্সিং ত্রুটির কারণে অনুপলব্ধ।",
                    "garment_industry_view": "বিশ্লেষণ পার্সিং ত্রুটির কারণে অনুপলব্ধ।",
                    "rural_leader_view": "বিশ্লেষণ পার্সিং ত্রুটির কারণে অনুপলব্ধ।"
                }

        except Exception as e:
            logger.error(f"Error generating expert summary: {e}")

            avg_happiness = step_stats.get('avg_happiness', 0.5)
            avg_support = step_stats.get('avg_support', 0.0)
            avg_income = step_stats.get('avg_income', 25000)
            policy_title = policy.get('title', 'policy')
            policy_domain = policy.get('domain', 'general')

            happiness_trend = "positive" if avg_happiness > 0.6 else "negative" if avg_happiness < 0.4 else "neutral"
            support_trend = "supportive" if avg_support > 0.2 else "opposed" if avg_support < -0.2 else "mixed"
            income_level = "above average" if avg_income > 30000 else "below average" if avg_income < 15000 else "moderate"

            return {
                "economist_view": f"The {policy_title} shows {happiness_trend} economic sentiment with average income at ৳{avg_income:,.0f}. Citizens are {support_trend} of this {policy_domain} policy. The {income_level} income levels suggest varying distributional effects across Bangladesh's economic strata, particularly impacting remittance-dependent families and the informal sector.",

                "activist_view": f"This {policy_domain} policy reveals concerning social dynamics. With {avg_happiness:.1%} average happiness and {support_trend} public opinion, marginalized communities — bosti dwellers, Hijra citizens, and religious minorities — face disproportionate impact. Stronger protective measures are needed for the most vulnerable Bangladeshis.",

                "garment_industry_view": f"From the RMG sector perspective, the {policy_title} creates {happiness_trend} conditions for Bangladesh's garment export industry. With average worker income at ৳{avg_income:,.0f}, any policy shift directly affects the 4 million+ garment workers and factory compliance. The {support_trend} sentiment signals potential labor market adjustments.",

                "rural_leader_view": f"For rural Bangladesh, the {policy_title} has {happiness_trend} implications. Agricultural livelihoods and fishing communities with ৳{avg_income:,.0f} average income face unique challenges from this {policy_domain} policy. The {support_trend} response from rural citizens highlights the need for localized implementation sensitive to flood-prone and char areas."
            }


def create_gemini_client(api_key: str, backup_keys: Optional[List[str]] = None) -> GeminiClient:
    """Create a GeminiClient instance with optional backup keys."""
    return GeminiClient(api_key, backup_keys)


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Client
# ─────────────────────────────────────────────────────────────────────────────

# Shared system prompts (same as GeminiClient, defined once to avoid duplication)
_CITIZEN_SYSTEM_PROMPT = """You are simulating how a fictional Bangladeshi citizen (নাগরিক) reacts to a new government policy or event.
You receive citizen_profile, current_state, and policy.

CRITICAL CONTEXT — BANGLADESH (বাংলাদেশ):
- Currency: BDT (৳ Bangladeshi Taka). All income values are MONTHLY in BDT. ৳1 USD ≈ ৳110 BDT.
- Income context: ৳8,000-15,000/month = low income (গরিব). ৳20,000-60,000 = middle class (মধ্যবিত্ত). ৳80,000+ = upper class (উচ্চবিত্ত).
- Economy: 7th largest by PPP in Asia. GDP growth ~6-7%. Inflation ~9%.
- Key sectors: Ready-Made Garments (RMG) employs 4M+ workers (80% women). Agriculture employs 40% of workforce. Remittance is 6% of GDP.
- Geography: Bangladesh is a low-lying delta, extremely vulnerable to floods, cyclones, and rising sea levels. Northern districts face drought.
- Social structure: Joint families common. 91% Muslim majority. Strong community bonds. Hierarchical respect structures.
- Urban reality: Dhaka is among the most densely populated cities (47,000 people/km²). bosti (slums) house ~40% of urban population.
- Rural reality: 63% of population is rural. Agriculture, fishing, and remittance are primary income sources.
- Digital: 130M+ internet users. bKash/Nagad mobile financial services widely used. Growing IT freelancing sector.

Output ONLY valid JSON with these exact keys:
- new_happiness: float between 0 and 1
- new_policy_support: float between -1 and 1
- income_delta: float in BDT (change in MONTHLY income, can be negative, zero, or positive)
- short_reason: string, 1-2 sentences explaining the reaction
- diary_entry: string, 3-5 sentences in first-person perspective as a Bangladeshi citizen

Do not include any explanation outside the JSON."""

_EXPERT_SYSTEM_PROMPT = """You are analyzing a synthetic Bangladesh society simulation.
Generate FOUR expert perspectives grounded in Bangladeshi socioeconomic reality.

1. অর্থনীতিবিদ (Economist): BDT income effects, inflation, GDP, remittance, employment.
2. সমাজকর্মী (Social Activist): poverty, bosti dwellers, garment workers, gender equity, NGO context (BRAC, Grameen).
3. গার্মেন্ট শিল্প প্রতিনিধি (Garment Industry): RMG sector, worker retention, export ($40B+), compliance.
4. গ্রামীণ নেতা (Rural Leader): agriculture, flood/cyclone resilience, food security, rural-urban migration.

Output ONLY valid JSON with keys:
- economist_view: string 3-5 sentences
- activist_view: string 3-5 sentences
- garment_industry_view: string 3-5 sentences
- rural_leader_view: string 3-5 sentences

No explanation outside JSON."""


class OllamaClient:
    """Local LLM client using Ollama — unlimited, no quota, no API keys needed."""

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")
        logger.info(f"OllamaClient ready — model={model}, host={host}")

    def _call(self, system: str, user: str, force_json: bool = True) -> str:
        """Send a chat request to Ollama and return response text."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user}
            ],
            "stream": False
        }
        if force_json:
            payload["format"] = "json"

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode())
                return body["message"]["content"]
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama connection failed — is 'ollama serve' running? ({e})") from e

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response — reuse same logic as GeminiClient."""
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(json_pattern, text, re.DOTALL):
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def generate_citizen_reaction(
        self,
        citizen_profile: Dict[str, Any],
        current_state: Dict[str, Any],
        policy: Dict[str, Any],
        knowledge_context: str = ""
    ) -> Dict[str, Any]:
        user_prompt = f"""Citizen Profile:\n{json.dumps(citizen_profile, indent=2)}\n\nCurrent State:\n{json.dumps(current_state, indent=2)}\n\nPolicy:\n{json.dumps(policy, indent=2)}\n\nGenerate the citizen's reaction as JSON."""

        system = _CITIZEN_SYSTEM_PROMPT
        if knowledge_context:
            # Inject web knowledge before the JSON output instruction
            knowledge_block = (
                "\n\nREAL-WORLD CONTEXT (from web search — use to ground your response):\n"
                + knowledge_context
                + "\n"
            )
            # Insert before "Output ONLY valid JSON"
            marker = "Output ONLY valid JSON"
            if marker in system:
                system = system.replace(marker, knowledge_block + "\n" + marker)
            else:
                system = system + knowledge_block

        text = self._call(system, user_prompt, force_json=True)
        result = self._extract_json(text)
        if not result:
            raise RuntimeError(f"Ollama returned unparseable JSON: {text[:200]}")

        return {
            "new_happiness":     max(0.0, min(1.0, float(result.get("new_happiness", 0.5)))),
            "new_policy_support": max(-1.0, min(1.0, float(result.get("new_policy_support", 0.0)))),
            "income_delta":      float(result.get("income_delta", 0.0)),
            "short_reason":      str(result.get("short_reason", "")),
            "diary_entry":       str(result.get("diary_entry", ""))
        }

    def generate_expert_summary(
        self,
        step_stats: Dict[str, Any],
        policy: Dict[str, Any]
    ) -> Dict[str, str]:
        user_prompt = f"""Policy:\n{json.dumps(policy, indent=2)}\n\nSimulation Statistics:\n{json.dumps(step_stats, indent=2)}\n\nGenerate expert perspectives as JSON."""

        text = self._call(_EXPERT_SYSTEM_PROMPT, user_prompt, force_json=True)
        result = self._extract_json(text)
        if not result:
            raise RuntimeError(f"Ollama returned unparseable JSON for expert summary: {text[:200]}")

        return {
            "economist_view":       str(result.get("economist_view", "")),
            "activist_view":        str(result.get("activist_view", "")),
            "garment_industry_view": str(result.get("garment_industry_view", "")),
            "rural_leader_view":    str(result.get("rural_leader_view", ""))
        }


def create_llm_client(
    backend: str = "ollama",
    ollama_model: str = "qwen2.5:7b",
    ollama_host: str = "http://localhost:11434",
    gemini_api_key: Optional[str] = None,
    backup_keys: Optional[List[str]] = None
):
    """
    Factory: return an OllamaClient or GeminiClient based on `backend`.

    Args:
        backend: "ollama" (local, unlimited) or "gemini" (cloud, quota-limited).
        ollama_model: Ollama model name, e.g. "qwen2.5:7b".
        ollama_host: Ollama server URL.
        gemini_api_key: Required when backend=="gemini".
        backup_keys: Gemini backup key list for rotation.
    """
    if backend == "ollama":
        return OllamaClient(model=ollama_model, host=ollama_host)
    elif backend == "gemini":
        if not gemini_api_key:
            raise ValueError("gemini_api_key is required when backend='gemini'")
        return GeminiClient(gemini_api_key, backup_keys)
    else:
        raise ValueError(f"Unknown LLM backend '{backend}'. Choose 'ollama' or 'gemini'.")
