<p align="center">
  <h1 align="center">NAGORIK-GENESIS</h1>
  <p align="center">
    <strong>Synthetic Society Policy Simulator — Bangladesh Edition</strong>
  </p>
  <p align="center">
    A hybrid AI system that simulates how 1,000–50,000 synthetic Bangladeshi citizens react to policy changes, combining LLM reasoning with neural network scalability.
  </p>
</p>

---

## The Problem

Bangladesh makes policy for **170 million people** with no way to pre-test the impact on diverse communities — garment workers, tech entrepreneurs, rural farmers, urban slum dwellers — before rolling it out nationwide.

## The Solution

NAGORIK-GENESIS generates a **synthetic population** mirroring Bangladesh's real demographics (8 divisions, income strata, professions, religions, political views) and simulates citizen reactions to any proposed policy using a three-engine hybrid AI approach.

---

## Features

- **Synthetic Population Engine** — Generates demographically accurate citizens across all 8 divisions with realistic income distributions (55% low / 35% middle / 10% high)
- **Three Simulation Modes**
  - `LLM_ONLY` — Every citizen processed by LLM (highest quality, slowest)
  - `NN_ONLY` — Pure neural network inference (fastest, needs training data)
  - `HYBRID` — Stratified LLM anchors + calibrated NN predictions (recommended)
- **Web Knowledge Enrichment** — Real-time policy context from Tavily / DuckDuckGo, summarized by the LLM before simulation
- **Expert Panel** — AI-generated analysis from 4 perspectives: Economist, Social Rights Activist, RMG Industry Analyst, and Rural Community Leader
- **8 Built-in Policy Presets** — Digital Taka (CBDC), Universal Basic Income, Climate Resilience Fund, RMG Minimum Wage Hike, EdTech for Madrasas, and more
- **Interactive Dashboard** — Streamlit UI with overview, demographic breakdowns, individual citizen diaries, and Plotly visualizations
- **Neural Network Learning** — MLP learns from LLM outputs across runs, continuously improving prediction quality
- **Scenario Comparison** — Run multiple policies side-by-side and compare outcomes

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Population  │  │  Simulation  │  │ Web Knowledge │  │
│  │  Generator   │  │   Engine     │  │   Pipeline    │  │
│  │ (population) │  │ (simulation) │  │(web_knowledge)│  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘  │
│         │                │                   │          │
│         ▼                ▼                   ▼          │
│  ┌─────────────────────────────────────────────────┐    │
│  │              LLM Client (Ollama / Gemini)       │    │
│  └─────────────────────────────────────────────────┘    │
│         │                │                              │
│         ▼                ▼                              │
│  ┌──────────────┐ ┌──────────────┐                     │
│  │  NN Model    │ │  ML Dataset  │                     │
│  │ (MLPRegress) │ │  (ml_data)   │                     │
│  └──────────────┘ └──────────────┘                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- **Python** 3.10+
- **Ollama** installed and running locally ([install guide](https://ollama.ai))

### 1. Clone & Install

```bash
git clone https://github.com/your-username/nagorik-genesis.git
cd nagorik-genesis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Up Ollama

```bash
ollama pull qwen2.5:7b
ollama serve  # if not already running
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Optional — enhances web knowledge quality (free fallback available)
TAVILY_API_KEY=your_tavily_api_key

# Only needed if using Gemini backend instead of Ollama
GEMINI_API_KEY=your_gemini_api_key
```

### 4. Launch

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `ollama` | LLM backend: `ollama` (local) or `gemini` (cloud) |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama model to use |
| `TAVILY_API_KEY` | — | Tavily API key for web knowledge (optional) |
| Population Size | 1,000 | Configurable up to 50,000 via UI slider |
| Simulation Steps | 5 | Number of time steps per simulation |
| NN Hidden Layers | (128, 64, 32) | MLP architecture for citizen reaction model |
| LLM Sample Size | 300 | Citizens processed by LLM per step in HYBRID mode |

---

## Project Structure

```
nagorik-genesis/
├── app.py                  # Streamlit entry point
├── simulation.py           # Core simulation engine (HYBRID/LLM/NN modes)
├── llm_client.py           # LLM clients (Ollama, Gemini)
├── population.py           # Synthetic population generator
├── nn_model.py             # Neural network (MLPRegressor wrapper)
├── ml_data.py              # Training data management
├── web_knowledge.py        # Web search + LLM summarization pipeline
├── data_models.py          # Core data structures (Citizen, Policy, etc.)
├── config.py               # Centralized configuration
├── utils.py                # Helpers, policy presets, feature engineering
├── stats.py                # Statistical analysis utilities
├── ui_sections.py          # Modular Streamlit UI components
├── batch_simulate.py       # CLI batch simulation runner
├── train_nn.py             # CLI neural network training
├── verify_simulation.py    # Simulation verification tool
├── test_nagorik.py         # Test suite
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── data/                   # Training data (CSV)
├── models/                 # Saved NN model + scalers (joblib)
└── knowledge_cache/        # Cached web knowledge results
```

---

## CLI Tools

```bash
# Run batch simulations across multiple policies
python batch_simulate.py

# Train the neural network from collected LLM data
python train_nn.py

# Verify simulation pipeline end-to-end
python verify_simulation.py

# Run tests
python test_nagorik.py
```

---

## How It Works

1. **Population Generation** — Creates synthetic citizens with 15+ attributes matching Bangladesh census distributions (division, income, religion, profession, education, political alignment, etc.)

2. **Web Knowledge Enrichment** — Before simulation, queries Tavily/DuckDuckGo for real-world context about the policy, then uses the LLM to summarize findings into structured bullet points.

3. **Simulation Loop** — For each time step:
   - **HYBRID mode** selects stratified LLM anchors (balanced across demographics) and processes them through the LLM for nuanced, narrative-rich reactions.
   - Remaining citizens are predicted by the calibrated neural network, bounded by LLM anchor ranges plus a tolerance margin.
   - Citizens accumulate state changes (happiness, income, policy support) with diary entries.

4. **Expert Analysis** — Four domain-specific AI experts analyze the aggregate results and provide policy recommendations.

5. **Neural Network Learning** — Each LLM-processed citizen becomes training data. The MLP (40-dimensional input → 3 outputs) learns to approximate LLM behavior, improving with each simulation run.

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit, Plotly |
| LLM | Ollama (qwen2.5:7b) / Google Gemini |
| ML Model | scikit-learn MLPRegressor |
| Web Search | Tavily API, DuckDuckGo |
| Data | pandas, NumPy |
| Serialization | joblib |

---

## Documentation

- [PROJECT_BLUEPRINT.md](PROJECT_BLUEPRINT.md) — Original design document
- [DEPLOYMENT_ROADMAP.md](DEPLOYMENT_ROADMAP.md) — Production deployment plan and limitations assessment
- [MODEL_SPECIFICATION.md](MODEL_SPECIFICATION.md) — Neural network architecture details
- [PRESENTATION.md](PRESENTATION.md) — Project presentation and speech notes
- [SIMULATION_ANALYSIS.md](SIMULATION_ANALYSIS.md) — Simulation quality analysis

---

## License

This project is developed for research and educational purposes.

---

<p align="center">
  Built for Bangladesh · Powered by Hybrid AI
</p>
