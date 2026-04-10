# NAGORIK-GENESIS

### *What if you could test a policy on 50,000 Bangladeshi citizens — before it touches a single real life?*

---

## What Is It?

NAGORIK-GENESIS is a **synthetic society simulator** for Bangladesh. You describe a policy — say, raising the garment minimum wage to BDT 18,000 — and it simulates how thousands of AI-generated citizens react: their happiness, income shifts, political support, and personal diary entries.

It's not a survey. It's not a spreadsheet model. It's a living, breathing digital Bangladesh where every citizen has a profession, a division, a religion, a family size, and an opinion.

---

## Why Bangladesh Needs This

Bangladesh makes policy for **170 million people** with almost no way to pre-test impact.

- A fuel subsidy cut hits a Dhaka rickshaw puller and a Rangpur farmer **completely differently** — but policy is designed as one-size-fits-all.
- The 2023 garment wage hike was celebrated — until factories in Gazipur started closing. **Nobody simulated the second-order effects.**
- Flood relief packages reach Barisal late every year. **Nobody models who actually gets left behind.**

NAGORIK-GENESIS lets policymakers **see the fractures before they happen** — across income levels, divisions, religions, and professions — in minutes, not months.

---

## How It Works

```
You write a policy
        ↓
   1,000+ synthetic citizens are generated
   (income, profession, division, religion, family, political view)
        ↓
   Each citizen reacts — powered by three engines:
```

| Engine | Role | Speed |
|--------|------|-------|
| **LLM** (Ollama/Qwen) | Deep, nuanced reactions with diary entries | ~2 sec/citizen |
| **Neural Network** | Learns from LLM, predicts at 10,000x speed | ~0.001 sec/citizen |
| **Rule-Based** | Deterministic fallback, always available | Instant |

**The trick:** the LLM teaches the neural network. Run 500 citizens through the LLM, train the NN, then simulate 50,000 in seconds. We call this **Hybrid AI distillation**.

Real-world context is injected via **web search** (Tavily/DuckDuckGo) — so the LLM knows about actual garment export data, current inflation rates, and recent policy debates before it responds.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      STREAMLIT UI                       │
│   Policy Input → Run Simulation → View Results          │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │    SIMULATION ENGINE    │
          │  (multi-step, per-citizen)│
          └──┬─────────┬─────────┬──┘
             │         │         │
     ┌───────▼──┐ ┌────▼────┐ ┌──▼───────┐
     │   LLM    │ │  NEURAL │ │  RULE    │
     │  ENGINE  │ │ NETWORK │ │  BASED   │
     │ (Ollama) │ │  (MLP)  │ │ FALLBACK │
     └───────┬──┘ └────┬────┘ └──┬───────┘
             │         │         │
     ┌───────▼─────────▼─────────▼────────┐
     │         CITIZEN POPULATION          │
     │  1,000 synthetic Bangladeshis       │
     │  (8 divisions × 5 zones × 3 income │
     │   × 8 professions × 5 religions)   │
     └───────────────┬────────────────────┘
                     │
     ┌───────────────▼────────────────────┐
     │        WEB KNOWLEDGE LAYER         │
     │  Tavily/DDG → Ollama Summarizer    │
     │  Real policy context injected      │
     └───────────────────────────────────┘
```

**Data flow:** Policy → Web search for real context → Stratified citizen sampling → LLM generates anchor reactions → NN predicts the rest → Aggregate stats by income/division/religion/zone → Expert panel summaries (Economist, Activist, RMG Analyst, Rural Leader)

---

## Current Limitations

| Area | Limitation |
|------|-----------|
| **Population** | Synthetic — not linked to real census or survey microdata |
| **LLM** | Storytelling, not causal modeling — no dose-response curves |
| **Neural Network** | Distills LLM biases, not ground truth |
| **Simulation** | No citizen memory, no social networks, no cascading effects |
| **Outputs** | Only 3 dimensions (happiness, support, income) |
| **Validation** | Never backtested against a real historical policy outcome |

---

## The Path to Real-World Deployment

| Phase | What Changes |
|-------|-------------|
| **1. Real Data** | Replace synthetic citizens with BBS/HIES microdata. Plug in Bangladesh Bank economic indicators. |
| **2. Smarter Models** | Fine-tune a 32B+ LLM on expert-annotated Bangladeshi policy reactions. Replace MLP with deep ensemble + uncertainty estimates. |
| **3. Richer Simulation** | Add social networks (family, workplace, community). Give citizens memory. Expand to 10+ output dimensions. |
| **4. Validate** | Backtest against 5 real historical policies. Achieve ≥80% directional accuracy. Bias audit across religion, gender, region. |

**The architecture is the skeleton. Real data is the muscle. Validation is the spine.**

---

> *NAGORIK-GENESIS doesn't predict the future. It stress-tests ideas — so that when a policy reaches the streets of Dhaka or the chars of Barisal, we've already seen what could go wrong.*
