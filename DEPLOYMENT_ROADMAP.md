# নাগরিক-GENESIS: Real-World Deployment Roadmap for Bangladesh

> **Purpose**: A brutally honest assessment of the current architecture's limitations, followed by a concrete engineering plan to make this system deployable for real policy analysis in Bangladesh — assuming no hardware constraints.

---

## Part 1: Honest Limitations of the Current Architecture

### 1.1 The Population Is Fiction, Not Data

**Current state**: `population.py` generates citizens from uniform/weighted random distributions. A "Garment Worker in Barisal earning ৳12,000" is assembled from `rng.choice()` — there is no real person, no real survey, no real behavioral data behind it.

**What this means**:
- The 55/35/10 income split is a rough BBS estimate, not linked to any census microdata
- Profession assignment is random within income brackets — a "Day Laborer" and "Garment Worker" have identical behavioral parameters except the label
- City zone assignment (`shohor_kendro`, `bosti`, `graam`) is uniform random, not correlated with division — but in reality, 40% of Dhaka is bosti while Rangpur is 90% rural
- `risk_tolerance` and `openness_to_change` are `rng.uniform(0, 1)` — pure noise with zero empirical basis
- `base_happiness` is `rng.uniform(0.3, 0.8)` — there's no data connecting Bangladeshi happiness to income, profession, division, or any demographic
- Family size distributions are invented, not from BDHS (Bangladesh Demographic and Health Survey)
- Education-income correlations are approximated, not from HIES (Household Income and Expenditure Survey)
- Political view is 4 categories assigned uniformly — real political alignment in Bangladesh is deeply correlated with region, income, profession, age, and religion

**Impact**: The population doesn't represent Bangladesh. It represents a randomized caricature of Bangladesh. Any aggregate result ("garment workers are 62% happy") is meaningless because the input atoms are meaningless.

### 1.2 The LLM Is Storytelling, Not Modeling

**Current state**: `llm_client.py` sends citizen profiles to qwen2.5:7b and asks it to produce `{new_happiness, new_policy_support, income_delta}` as JSON.

**What this means**:
- **No causal model**: The LLM doesn't know that a 56% garment wage hike in 2023 led to 12% factory closures in Gazipur. It generates plausible-sounding reactions from its training data, which is mostly English-language internet text about South Asia — not Bangladeshi microeconomic data
- **No temporal consistency**: Two identical citizens with the same profile can get wildly different reactions on consecutive calls. There's no mechanism to ensure a citizen's trajectory is internally coherent across steps
- **Anchoring bias**: The system prompt says "৳8,000-15,000 = poor" which anchors ALL low-income responses to a narrow band, regardless of the actual citizen profile
- **No counterfactual reasoning**: The LLM can't answer "what would happen if the wage were ৳16,000 instead of ৳18,000" — it doesn't have a dose-response model
- **Hallucination risk**: The `web_knowledge.py` context helps, but the LLM can blend real facts with invented statistics seamlessly
- **Language blindness**: qwen2.5:7b has limited Bangla understanding. The nuances of how a Noakhali fisherman thinks differently from a Rajshahi farmer are largely invisible to the model
- **JSON output mode forces shallow reasoning**: The `force_json=True` flag in Ollama skips chain-of-thought. The model jumps directly to numbers

### 1.3 The Neural Network Learns Noise

**Current state**: `nn_model.py` is an MLPRegressor(128, 64, 32) that learns `[Δhappiness, Δsupport, Δincome]` from LLM-generated samples.

**What this means**:
- The NN is a **distillation of the LLM's biases**, not a model of reality. If the LLM consistently overestimates garment worker happiness, the NN will too
- 40-dimensional feature space with one-hot encoding creates sparse inputs — many (income_level × division × religion) intersections have near-zero training samples
- A single MLP trained across ALL policy domains means healthcare-domain patterns are diluted by economy-domain patterns. The model cannot specialize
- StandardScaler on targets means income deltas (range ±500,000 BDT) and happiness deltas (range ±1.0) are mixed — income variance dominates
- No epistemic uncertainty: the model outputs a point estimate with no confidence interval. A prediction for a "Buddhist IT Professional in Mymensingh" (extremely rare cell) gets the same confidence as a "Muslim Garment Worker in Dhaka" (most common cell)

### 1.4 The Simulation Is Memoryless

**Current state**: `simulation.py` computes each step independently. A citizen's step-2 state depends only on step-1 state + policy + profile.

**What this means**:
- No adaptation: citizens don't learn, protest, migrate, or form groups
- No network effects: a garment factory closure affecting 2,000 workers in Ashulia doesn't cascade to the tea stall owners who serve them
- No temporal dynamics: inflation compounds, debt accumulates, seasonal cycles (boro rice harvest, monsoon floods, Eid spending) don't exist
- No policy feedback: if 80% of citizens oppose a policy, nothing changes — the policy is immutable for all steps
- No spatial dynamics: a policy that builds a bridge in Padma doesn't affect Barisal and Dhaka differently over time

### 1.5 Three Outputs Are Not Enough

**Current state**: The simulation tracks 3 outputs: happiness (0-1), policy_support (-1 to +1), income (BDT).

**What this means**:
- Missing: health status, employment status (employed/unemployed/underemployed), food security, child school enrollment, migration intent, debt level, social cohesion, trust in institutions
- A healthcare policy that bankrupts a family but keeps them "happy" (because they got treated) is indistinguishable from one that both heals and doesn't bankrupt
- Income is a single number — no distinction between formal wage, informal wage, remittance, agricultural income, and transfers

### 1.6 Web Knowledge Is Shallow

**Current state**: `web_knowledge.py` searches Tavily/DDG, summarizes 15 snippets into 8-15 bullets via Ollama.

**What this means**:
- Web search returns news articles and opinion pieces, not structured economic data
- No access to Bangladesh Bureau of Statistics (BBS) microdata, HIES tables, or World Bank indicators
- Summarization is lossy — key numbers get dropped or hallucinated during the Ollama consolidation
- Cache keyed by (title, domain) only — different policy *descriptions* with the same title hit the same cache

### 1.7 Validation Is Nonexistent

**Current state**: There's no mechanism to check if simulation outputs match any known real-world outcome.

**What this means**:
- We cannot tell if the system produces better predictions than a random number generator
- No backtesting: we've never simulated a *past* policy (e.g., 2023 garment wage hike) and compared the output to what actually happened
- No calibration: we don't know if "happiness = 0.62" means anything

---

## Part 2: Real-World Deployment Architecture

### 2.1 Data Source Overhaul

#### Replace Synthetic Population with Survey Microdata

| Current | Target | Source |
|---------|--------|--------|
| `rng.uniform(8000, 15000)` for low income | Actual income distributions from HIES unit records | Bangladesh Bureau of Statistics — HIES 2022 |
| `rng.choice(PROFESSIONS)` random | Labour Force Survey occupation codes mapped to ISCO-08 | BBS Quarterly Labour Force Survey |
| Uniform `risk_tolerance` | Derived from consumption volatility in panel data | World Bank BIHS (Bangladesh Integrated Household Survey) |
| Random `base_happiness` | Well-being module from national survey | BIDS Household Survey / Gallup World Poll Bangladesh |
| Random education-income link | Actual joint distribution from census | BBS Population Census 2022 microdata |
| 4 political views, uniform | Political preference surveys by division | IRI/NDI Bangladesh public opinion polls |
| Random city zone | GIS-tagged settlement classification | BBS Urban Area Directory + LGED maps |

**Pipeline change**: Replace `generate_population()` with a `load_population_from_microdata()` function that reads cleaned survey files (Parquet/CSV) and maps real household records → `Citizen` objects. For privacy, apply k-anonymity (k≥5) on the loaded records. For scaling to 1M+ citizens, use statistical bootstrapping from the microdata with controlled noise injection.

#### Real-Time Economic Indicators

| Indicator | Source | Refresh |
|-----------|--------|---------|
| CPI / Inflation | Bangladesh Bank Monthly Economic Trends | Monthly |
| Remittance inflow | Bangladesh Bank (BEPB) | Monthly |
| RMG export volume | BGMEA/BKMEA export data | Monthly |
| Rice / essential commodity prices | DAM (Directorate of Agricultural Marketing) | Weekly |
| Employment statistics | BBS Labour Force Survey | Quarterly |
| Mobile financial services volume | Bangladesh Bank MFS data | Monthly |
| Government budget allocations | Ministry of Finance / iBAS++ | Annual |

**Pipeline change**: Create an `EconomicContext` service that fetches, caches, and exposes these indicators. Inject as structured data (not free-text bullets) into both LLM prompts and feature vectors.

#### Administrative & Geo-Spatial Data

| Data | Source | Use |
|------|--------|-----|
| Upazila-level population density | BBS | Spatial heterogeneity |
| Flood risk zones | BWDB flood danger maps | Climate policy impact |
| Healthcare facility density | DGHS facility registry | Healthcare access modeling |
| School density by upazila | BANBEIS | Education policy reach |
| Road connectivity index | LGED | Infrastructure policy impact |
| Factory locations (RMG) | BGMEA member list with GPS | Industrial policy modeling |

**Pipeline change**: Add a `GeoContext` layer that maps each citizen's division + upazila to physical infrastructure scores. These become additional feature dimensions for the NN.

### 2.2 LLM Pipeline Restructure

#### Fine-Tuned Bangladesh Policy Model

**Current**: Generic qwen2.5:7b with a long system prompt.

**Target**: Fine-tuned model on Bangladesh-specific policy reaction data.

**Training data sources for fine-tuning**:
1. **Historical policy reactions** (manually curated):
   - 2023 garment wage hike: news articles → structured reaction labels by demographic group
   - 2022 fuel price hike: actual protest data, consumer spending changes
   - 2024 Digital Bangladesh initiatives: MFS adoption rates by division
   - COVID-19 stimulus packages: disbursement and consumption data
   - Padma Bridge opening: transport cost changes, regional GDP impact
   
2. **Expert-annotated synthetic reactions**:
   - Commission Bangladeshi economists, sociologists, labor experts to review and correct LLM-generated reactions
   - Create 10,000+ validated (citizen_profile, policy, reaction) tuples
   - Use DPO (Direct Preference Optimization) to align the model with expert judgments

3. **Survey-based reaction data**:
   - Partner with BIDS/BBS to run public opinion surveys on past policy changes
   - Map survey responses to the (happiness, support, income_delta) schema
   - Use as gold-standard validation set

**Fine-tuning approach**:
```
Base model: Qwen2.5-32B or LLaMA-3.1-70B (no hardware constraints)
Method: QLoRA fine-tuning → Full fine-tune → DPO alignment
Training set: 50,000+ (profile, policy, reaction) tuples
Validation: Hold-out set of expert-annotated reactions
Infrastructure: 4× A100 80GB or 2× H100 for 70B model
```

#### Structured Reasoning Before JSON

**Current**: `force_json=True` → model jumps to numbers.

**Target**: Two-stage generation.

```
Stage 1 (thinking): "Think step-by-step about how this citizen would react..."
  → Free-text chain of thought (not shown to user, not parsed)
  
Stage 2 (structured): "Now output your final answer as JSON..."
  → Parse JSON from second generation
```

This lets the model reason about profession-policy interaction, division-specific context, and temporal dynamics before committing to numbers. The thinking trace also becomes an audit trail.

#### Multi-LLM Consensus

For high-stakes policy analysis, don't trust a single LLM call:

```
                    ┌─── LLaMA-70B ────┐
Citizen Profile ───►├─── Qwen2.5-32B ──├──► Median Aggregation ──► Final Reaction
                    └─── Gemini Flash ──┘
```

Take the **median** of 3 models' outputs. If any model's output deviates >2σ from others, flag that citizen for human review. This eliminates single-model hallucination.

### 2.3 Neural Network Architecture Overhaul

#### Replace MLP with Specialized Architecture

**Current**: Single MLPRegressor(128, 64, 32) for all domains.

**Target**: Domain-specific ensemble with uncertainty estimation.

```
Architecture: Policy-Conditioned Deep Ensemble
─────────────────────────────────────────────
Input (citizen features + economic context + geo features):
  │
  ├── Shared Embedding Layer (citizen demographics → 64-dim)
  │
  ├── Policy Encoder (policy text → 128-dim via sentence transformer)
  │
  ├── Economic Context Encoder (macro indicators → 32-dim)
  │
  ├── Geo-Spatial Encoder (upazila features → 32-dim)
  │
  └── Concatenate → 256-dim
        │
        ├── Ensemble Member 1: MLP(256, 128, 64) → [Δh, Δs, Δi]
        ├── Ensemble Member 2: MLP(256, 128, 64) → [Δh, Δs, Δi]
        ├── Ensemble Member 3: MLP(256, 128, 64) → [Δh, Δs, Δi]
        ├── Ensemble Member 4: MLP(256, 128, 64) → [Δh, Δs, Δi]
        └── Ensemble Member 5: MLP(256, 128, 64) → [Δh, Δs, Δi]
              │
              └── Output: mean prediction + standard deviation (epistemic uncertainty)
```

**Key improvements**:
- **Deep Ensemble** (5 members): each trained with different random seeds → disagreement = uncertainty
- **Policy encoder**: encode policy *text* as embedding (not just domain one-hot), so "Raise garment wage to ৳18,000" and "Raise garment wage to ৳25,000" produce different embeddings
- **Macro context encoder**: real-time CPI, remittance flow, rice price feed directly into the prediction — the model can learn that a wage hike during 9% inflation is different from one during 4% inflation
- **Uncertainty output**: instead of point estimate, output `(mean, std)` — high uncertainty triggers LLM fallback even in NN_ONLY mode

**Framework**: PyTorch (not scikit-learn). Train on GPU. Use PyTorch Lightning for distributed training across multiple GPUs if the validated dataset grows to millions.

#### Expand Output Space

**Current**: 3 outputs `[Δhappiness, Δsupport, Δincome]`

**Target**: 10+ outputs capturing real welfare dimensions.

| Output | Type | Range | Description |
|--------|------|-------|-------------|
| Δhappiness | float | [-1, 1] | Subjective well-being change |
| Δpolicy_support | float | [-2, 2] | Political stance shift |
| Δincome_formal | float | BDT | Formal wage/salary change |
| Δincome_informal | float | BDT | Informal/gig income change |
| Δincome_remittance | float | BDT | Remittance income change |
| Δhealth_expenditure | float | BDT | Out-of-pocket health spending change |
| Δfood_security | float | [0, 1] | Food security index change |
| employment_status_change | categorical | {none, lost_job, found_job, underemployed} | Employment transition |
| migration_intent | categorical | {none, urban, abroad} | Migration pressure |
| protest_probability | float | [0, 1] | Likelihood of participating in protest/hartal |
| Δdebt | float | BDT | Household debt change |
| Δchild_enrollment | float | [0, 1] | Probability of children remaining in school |

### 2.4 Simulation Engine Restructure

#### Agent-Based Model with Networks

**Current**: Each citizen is updated independently. No interaction.

**Target**: Citizens exist in a social network with influence propagation.

```
Network Layers:
─────────────────
1. Family Network: household members share income shocks (100%)
2. Workplace Network: colleagues in same profession × division share employment effects (70%)
3. Community Network: same city_zone × division share sentiment (40%)
4. Information Network: connected to media/social media with varying exposure (varies by education, age, digital access)
```

**Interaction model per step**:
```
For each step:
  1. Compute direct policy impact (LLM/NN) → individual reaction
  2. Propagate through family network → shared income shock
  3. Propagate through workplace network → employment cascade
  4. Propagate through community network → sentiment diffusion
  5. Apply information network → media influence on support
  6. Resolve conflicts (a citizen who lost their job but whose neighbor got a new job)
  7. Update state
```

**Why this matters**: When a garment factory in Gazipur lays off 500 workers due to a wage hike, the current system only affects those 500 citizens. In reality, the impact cascades: tea stall owners lose customers, rickshaw pullers lose riders, landlords lose rent, children drop out of school. This network captures those second and third-order effects.

#### Temporal Dynamics and Memory

**Current**: Stateless — each step is independent.

**Target**: Citizens accumulate memory and adapt.

```python
@dataclass
class CitizenMemory:
    """Persistent state that evolves across simulation steps."""
    debt_accumulated: float = 0.0
    months_unemployed: int = 0
    consecutive_negative_shocks: int = 0
    has_migrated: bool = False
    protest_history: List[int] = field(default_factory=list)
    trust_in_government: float = 0.5  # decays with broken promises
    adaptation_factor: float = 1.0    # diminishing sensitivity to repeated same-type policies
```

**Behavioral rules from memory**:
- **Desperation threshold**: If `consecutive_negative_shocks > 3` AND `debt > 6 × income`, trigger migration or protest (not just happiness drop)
- **Adaptation**: Repeated exposure to same policy domain reduces reaction magnitude (people get used to it)
- **Trust decay**: If a policy promised income increase but delivered decrease, future government policies get lower initial support
- **Hysteresis**: A citizen who lost their job doesn't instantly recover when a pro-employment policy arrives — recovery takes multiple steps

#### Multi-Step Policy Simulation

**Current**: A single policy runs for N steps.

**Target**: Policy sequencing with interactions.

```
Example scenario:
  Step 1-3: "Garment Minimum Wage Increase to ৳18,000"
  Step 4-6: "Export Subsidy for RMG Factories"
  Step 7-9: "Skill Training Program for Displaced Workers"

The system simulates how Policy 2 interacts with the aftermath of Policy 1,
and how Policy 3 addresses the displacement caused by both.
```

Each policy modifies the economic context that subsequent policies operate in. A wage hike that causes factory closures changes the employment landscape for the export subsidy policy.

#### Seasonal and Event Calendar

Bangladesh has predictable seasonal patterns that massively affect citizen reactions:

| Month | Event | Effect |
|-------|-------|--------|
| Apr-May | Boro rice harvest | Rural income spike |
| Jun-Sep | Monsoon + floods | Agricultural loss, urban waterlogging, construction halt |
| Aug-Oct | Eid-ul-Adha + Durga Puja | Consumption spike, garment demand up |
| Nov-Feb | Dry season | Construction boom, brick kiln activation |
| Mar-Apr | Pre-monsoon cyclone risk | Coastal vulnerability peak |

The simulation should inject seasonal modifiers that shift baseline economic conditions before applying policy effects.

### 2.5 Citizen Caching & Token Optimization

#### Citizen Archetype Caching

**Current insight you identified**: Every LLM call sends the full citizen profile + state + policy. For 500 citizens × 5 steps = 2,500 calls — massive token waste.

**Solution: Archetype-based caching with residual correction.**

```
Step 1: Cluster citizens into archetypes
────────────────────────────────────────
  Archetype A: Low-income, Garment Worker, Dhaka, Female, Age 20-30, Muslim
  Archetype B: Middle-income, Teacher, Rajshahi, Male, Age 35-50, Muslim  
  Archetype C: Low-income, Farmer, Barisal, Male, Age 40-60, Muslim, Remittance
  ... (50-100 archetypes covering 95% of population)

Step 2: One LLM call per archetype per step
────────────────────────────────────────────
  Instead of 500 individual calls → ~80 archetype calls
  Each archetype gets a detailed, representative LLM reaction

Step 3: Residual correction for individual citizens
────────────────────────────────────────────────────
  For citizen in archetype A:
    base_reaction = archetype_A_reaction
    individual_correction = lightweight_NN(citizen_delta_from_archetype_center)
    final_reaction = base_reaction + individual_correction
```

**Token savings**: 80 LLM calls instead of 500 = **84% reduction**. With larger populations (50,000 citizens), savings are 99.8%.

#### Prompt Compression

**Current**: Full citizen profile JSON is ~500 tokens per call.

**Optimization**:
```
Current (verbose):
{
  "id": 42, "age": 28, "gender": "Female", "city_zone": "bosti",
  "income_level": "low", "education": "SSC", "profession": "Garment Worker",
  "family_size": 5, "political_view": "neutral", "risk_tolerance": 0.62,
  "openness_to_change": 0.45, "base_happiness": 0.48, "base_income": 12500,
  "division": "Dhaka", "religion": "Muslim", "is_remittance_family": false
}

Compressed (~60% fewer tokens):
F28|bosti|low|SSC|GarmentWorker|fam5|neutral|r0.62|o0.45|h0.48|i12500|Dhaka|Muslim|noRemit
```

Train the fine-tuned model to accept this compressed format. At 60% token reduction per citizen, costs drop proportionally.

### 2.6 Validation Framework

This is what's **completely missing** and **most critical** for real-world credibility.

#### Backtesting Against Historical Policies

| Policy | Date | Ground Truth Source | Metrics to Compare |
|--------|------|--------------------|--------------------|
| Garment wage hike ৳8,000→৳12,500 | Nov 2023 | BBS labor survey, news reports on factory closures | Employment %, income change, protest incidence |
| Fuel price increase (diesel +42%) | Aug 2022 | Transport fare data, CPI, protest reports | Transport cost, inflation pass-through, public sentiment |
| COVID-19 cash transfer (৳2,500) | Apr 2020 | World Bank evaluation, G2P disbursement data | Consumption change, food security |
| Padma Bridge toll introduction | Jun 2022 | Transport cost surveys, regional GDP estimates | Division-level economic impact |
| Digital Bangladesh ID rollout | 2023-24 | NID registration stats, MFS adoption data | Digital inclusion by division, age group |

**Process**:
1. Configure simulation with pre-policy population (from HIES data closest to policy date)
2. Run simulation with historical policy parameters
3. Compare simulation outputs against actual measured outcomes
4. Compute prediction error metrics (MAE, RMSE, direction accuracy)
5. Iterate: re-tune LLM prompts, NN architecture, rule-based parameters until error is acceptable

**Acceptance criteria**: The simulation should predict the *direction* of impact correctly for ≥80% of demographic subgroups, and the magnitude should be within 1 standard deviation of actual survey data.

#### Continuous Calibration

Once deployed, every real policy event becomes a calibration opportunity:

```
Real Policy Announced → Run Simulation → Compare with actual survey data (3-6 months later)
                                              │
                                              ▼
                                    Calibration Error
                                              │
                                    ┌─────────┴─────────┐
                                    │                     │
                              Error < threshold      Error > threshold
                                    │                     │
                                    ▼                     ▼
                              Log & continue        Trigger re-training
                                                   (fine-tune LLM, retrain NN)
```

### 2.7 Infrastructure for Deployment

#### Compute Architecture (No Hardware Constraints)

```
┌──────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT ARCHITECTURE                     │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────┐    ┌──────────────────┐                  │
│  │   Streamlit /    │    │   FastAPI Backend │                  │
│  │   Next.js UI     │◄──►│   (async workers) │                  │
│  └─────────────────┘    └────────┬─────────┘                  │
│                                   │                             │
│           ┌───────────────────────┼───────────────────┐        │
│           │                       │                    │        │
│  ┌────────▼────────┐  ┌─────────▼─────────┐  ┌──────▼──────┐ │
│  │  LLM Cluster     │  │  NN Inference      │  │  Data Store  │ │
│  │  (vLLM / TGI)    │  │  (TorchServe)      │  │  (Postgres   │ │
│  │                   │  │                     │  │   + Redis)   │ │
│  │  4× A100 80GB    │  │  2× T4 / L4        │  │              │ │
│  │  LLaMA-70B       │  │  Deep Ensemble ×5   │  │  Microdata   │ │
│  │  + Qwen-32B      │  │  + Policy Encoder   │  │  Indicators  │ │
│  │  + Gemini (API)   │  │                     │  │  Results     │ │
│  └──────────────────┘  └─────────────────────┘  └─────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Data Pipeline (Airflow)                  │  │
│  │  • BBS microdata ingestion (quarterly)                     │  │
│  │  • Bangladesh Bank indicators (monthly)                    │  │
│  │  • Web knowledge refresh (weekly)                          │  │
│  │  • Model retraining trigger (on calibration error)         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Monitoring & Evaluation                  │  │
│  │  • Prediction vs. reality dashboard                        │  │
│  │  • LLM drift detection (output distribution shift)         │  │
│  │  • Bias auditing (differential impact by religion/gender)  │  │
│  │  • Cost tracking (LLM tokens, GPU hours)                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

#### LLM Serving

- **vLLM** for local model serving (continuous batching, PagedAttention)
- 70B model on 4× A100 80GB with tensor parallelism
- Expected throughput: ~200 tokens/sec for 70B → ~50 citizen reactions/minute
- For 50,000 citizens with archetype caching (100 archetypes): ~2 minutes per step

#### Database

Replace in-memory lists with PostgreSQL:
- Citizens table with survey-linked demographic data
- Simulation runs table with versioned results
- Time-series state table (partitioned by step)
- Economic indicators table (time-series)
- Audit trail: every LLM call logged with input, output, latency, model version

Redis for:
- Archetype cache (LLM reactions per archetype per policy)
- Rate limiting
- Session state (replace Streamlit session state)

### 2.8 Ethical and Governance Requirements

#### Bias Auditing

Before any deployment, the system MUST be audited for:

1. **Religious bias**: Do Hindu citizens systematically get different outcomes from Muslim citizens for the same policy? If so, is it justified by data or is it LLM bias?
2. **Gender bias**: Does the model underestimate income effects for women because the LLM has patriarchal biases from training data?
3. **Regional bias**: Do Dhaka citizens always get better outcomes because the LLM has more training data about Dhaka?
4. **Hijra representation**: The 0.3% Hijra population has almost zero representation in training data — their predictions are essentially random

**Mitigation**: Stratified evaluation (compute metrics separately per demographic group), adversarial testing (inject counterfactual citizens that differ only in religion/gender and check for differential treatment), and human review of 500+ diary entries for stereotyping.

#### Responsible Use Policy

- Results MUST be labeled as "exploratory simulation, not prediction"
- No use for individual-level decisions (e.g., "this citizen type will suffer" → deny services)
- Government users must understand epistemic uncertainty — display confidence intervals, not point estimates
- Raw diary entries must not be used to characterize real demographic groups
- External review board (Bangladeshi academics + civil society) must sign off before public deployment

---

## Part 3: Implementation Priority (Phased Roadmap)

### Phase 1: Foundation (Months 1-3)
- [ ] Replace synthetic population with HIES 2022 microdata
- [ ] Build backtesting framework with 5 historical policies
- [ ] Add epistemic uncertainty to NN (Deep Ensemble, 5 members, PyTorch)
- [ ] Implement citizen archetype caching (80% LLM token savings)
- [ ] Database migration (PostgreSQL + Redis)

### Phase 2: Data Enrichment (Months 3-6)
- [ ] Integrate real-time Bangladesh Bank economic indicators
- [ ] Add geo-spatial features (flood risk, infrastructure density)
- [ ] Commission expert annotation of 10,000 policy reactions
- [ ] Expand output space to 10+ welfare dimensions
- [ ] Seasonal calendar injection

### Phase 3: Model Upgrade (Months 6-9)
- [ ] Fine-tune 32B/70B model on Bangladesh policy data (QLoRA → DPO)
- [ ] Implement two-stage reasoning (think → JSON)
- [ ] Multi-LLM consensus for critical predictions
- [ ] Agent-based network simulation (family, workplace, community layers)
- [ ] Citizen memory and adaptation

### Phase 4: Validation & Deployment (Months 9-12)
- [ ] Backtest all 5 historical policies, achieve ≥80% direction accuracy
- [ ] Bias audit (religion, gender, region, Hijra)
- [ ] External review board sign-off
- [ ] Production infrastructure (vLLM, TorchServe, Airflow)
- [ ] Continuous calibration pipeline

---

## Part 4: What Stays Good

Not everything is broken. Credit where due:

1. **The hybrid architecture (LLM → NN distillation → calibrated prediction)** is fundamentally sound. The core idea of using LLM for rich micro-simulation and NN for scalability is exactly what real agent-based computational social science papers propose. The problem isn't the architecture — it's the data feeding it.

2. **Stratified anchor selection** is a genuine improvement over random sampling. This is a legitimate variance-reduction technique from survey methodology.

3. **Web knowledge injection** is ahead of most academic systems. Real-time grounding of LLM prompts with web-sourced evidence is a practical and valuable innovation.

4. **The 40-dimensional feature space** covers the right Bangladesh-specific variables (division, religion, remittance, city_zone). The encoding is correct — it just needs real data behind it.

5. **The Streamlit interface** with Bangla localization makes the tool accessible to Bangladeshi policymakers, which is the entire point.

**The gap is not in the code — it's between the code and reality.** The architecture is a solid skeleton. Real-world deployment is about replacing the synthetic organs with real ones.
