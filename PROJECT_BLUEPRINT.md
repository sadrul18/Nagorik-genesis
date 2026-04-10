# Nagorik Genesis: Implementation Plan

> Bangladesh Policy Simulator — a synthetic society platform powered by hybrid AI.

---

## Table of Contents

1. [Vision](#1-vision)
2. [Product Goals](#2-product-goals)
3. [Core System Design](#3-core-system-design)
4. [Implementation Roadmap](#4-implementation-roadmap)
5. [Milestone Plan](#5-milestone-plan)
6. [Data & Model Strategy](#6-data--model-strategy)
7. [Testing Strategy](#7-testing-strategy)
8. [Launch Plan](#8-launch-plan)
9. [Future Direction](#9-future-direction)

---

## 1. Vision

Nagorik Genesis will become a Bangladesh-focused synthetic society simulator for exploring how policies may affect different groups of citizens over time.

The platform will help users test policy ideas, observe simulated citizen reactions, compare demographic impact, and scale analysis through a hybrid AI pipeline that combines LLM reasoning, neural-network inference, and deterministic rule-based fallback.

| Layer | Purpose |
|-------|---------|
| Synthetic population | Will generate realistic Bangladeshi citizen profiles |
| Policy simulation | Will model citizen reactions across multiple time steps |
| LLM reasoning | Will generate nuanced individual reactions and diary entries |
| Neural network | Will learn from LLM reactions and scale simulations quickly |
| Rule engine | Will keep simulations running without API dependency |
| Analytics dashboard | Will expose trends, group differences, and scenario comparisons |

---

## 2. Product Goals

| Goal | Outcome |
|------|---------|
| Simulate Bangladesh policy impact | Users will test policies across income, geography, religion, profession, and remittance status |
| Support multiple simulation modes | LLM_ONLY, HYBRID, and NN_ONLY modes will support accuracy, balance, and scale |
| Make results explainable | Citizen diaries and expert views will make aggregate numbers easier to interpret |
| Scale beyond manual LLM calls | Neural-network inference will allow large population runs without high API usage |
| Stay usable with limited resources | Ollama, Gemini, and rule-based fallback will support multiple runtime options |
| Keep outputs responsible | The product will present results as synthetic scenario exploration, not real-world prediction |

---

## 3. Core System Design

```text
Nagorik Genesis
├── Streamlit application
│   ├── Sidebar controls
│   ├── Simulation dashboard
│   ├── Citizen browser
│   ├── Expert perspectives
│   └── Scenario comparison
├── Simulation core
│   ├── Population generation
│   ├── Policy configuration
│   ├── LLM / NN / rule routing
│   └── Step-by-step state updates
├── AI layer
│   ├── Gemini adapter
│   ├── Ollama adapter
│   ├── JSON validation
│   └── fallback summaries
├── ML layer
│   ├── Training sample dataset
│   ├── Feature engineering
│   ├── MLP reaction model
│   └── model + scaler persistence
└── Analytics layer
    ├── Time-series metrics
    ├── Income group analysis
    ├── Zone and division analysis
    ├── Religion and remittance analysis
    └── scenario comparison
```

| Component | Responsibility |
|-----------|----------------|
| Data models | Will define citizens, policies, simulation config, citizen states, and aggregate stats |
| Population generator | Will create seeded Bangladeshi citizen profiles with realistic distributions |
| Simulation engine | Will run multi-step policy simulations and choose the update method per citizen |
| LLM client | Will generate structured citizen reactions and expert summaries |
| ML dataset manager | Will save, load, and merge training samples |
| Neural network model | Will train on LLM outputs and predict citizen reaction deltas |
| Statistics module | Will compute population and group-level metrics |
| UI modules | Will render controls, charts, citizen details, expert views, and scenario tools |

---

## 4. Implementation Roadmap

| Phase | Focus | Result |
|-------|-------|--------|
| Phase 1 | Product foundation | Project structure, configuration, domain constants, and dataclasses |
| Phase 2 | Bangladesh population model | Realistic citizen generation with BDT income and local demographics |
| Phase 3 | Simulation engine | Multi-step simulation with rule-based, LLM, and NN update paths |
| Phase 4 | LLM integration | Gemini and Ollama support with structured JSON responses |
| Phase 5 | ML pipeline | Training dataset, feature vectors, model training, and scaler-safe inference |
| Phase 6 | Analytics dashboard | Streamlit UI with charts, filters, citizens, experts, and scenarios |
| Phase 7 | Verification | Unit tests, simulation checks, ML smoke tests, and UI import checks |
| Phase 8 | Launch readiness | Local setup, cloud configuration, documentation, and release checklist |

---

## 5. Milestone Plan

### Phase 1 — Product Foundation

Nagorik Genesis will begin with a clear product contract and a stable module layout.

| Task | Deliverable |
|------|-------------|
| Define application settings | Runtime configuration for LLM backend, API keys, population limits, random seed, and NN parameters |
| Define core dataclasses | Citizen, CitizenState, PolicyInput, SimulationConfig, StepStats, ExpertSummary |
| Define domain constants | Policy domains, city zones, income levels, political views, genders, education levels, divisions, and religions |
| Establish project commands | Local run, test, batch simulation, and training commands |
| Add responsible AI copy | Clear positioning as synthetic scenario exploration |

**Definition of done**: the project will have a typed foundation that every later layer can depend on.

### Phase 2 — Bangladesh Population Model

The population system will generate synthetic citizens grounded in Bangladesh context.

| Attribute | Planned Model |
|-----------|---------------|
| Income | Monthly BDT income ranges for low, middle, and high-income groups |
| Geography | Dhaka, Chittagong, Rajshahi, Khulna, Barisal, Sylhet, Rangpur, Mymensingh |
| City zone | `shohor_kendro`, `shilpo_elaka`, `uposhohon`, `graam`, `bosti` |
| Gender | Male, Female, Hijra |
| Religion | Muslim, Hindu, Buddhist, Christian, Other |
| Family profile | Zone-sensitive family size and remittance-family flag |
| Education | No Formal Education, Madrasa Education, SSC, HSC, Honours/Bachelor's, Masters, PhD |
| Profession | Income-aware professions such as garment worker, farmer, teacher, NGO worker, senior doctor, and factory owner |

**Definition of done**: seeded population generation will produce valid, reproducible, Bangladesh-aware citizen profiles.

### Phase 3 — Simulation Engine

The simulation engine will update every citizen over one or more time steps.

| Task | Deliverable |
|------|-------------|
| Initialize citizen states | Step 0 state from base happiness, income, and neutral policy support |
| Run policy steps | Step-by-step updates for all citizens |
| Route update method | LLM_ONLY, HYBRID, NN_ONLY, and rule-based fallback |
| Maintain fast state lookup | Efficient access to previous citizen state |
| Clamp state values | Happiness `[0,1]`, support `[-1,1]`, income `>=0` |
| Track method usage | Per-step counts for LLM, NN, and rule-based updates |

**Definition of done**: the simulator will run small LLM scenarios and large NN/rule-based scenarios with stable state transitions.

### Phase 4 — LLM Integration

The LLM layer will support cloud and local generation.

| Backend | Role |
|---------|------|
| Gemini | Cloud LLM backend for high-quality structured responses |
| Ollama | Local LLM backend for unlimited development and batch generation |

| Task | Deliverable |
|------|-------------|
| Create backend factory | A single client factory will select Gemini or Ollama |
| Design citizen prompt | Bangladesh-aware policy reaction prompt with strict JSON schema |
| Design expert prompt | Economist, social activist, garment industry, and rural leader perspectives |
| Validate JSON output | Numeric clamping and fallback values for malformed responses |
| Handle quota and failures | Rate limiting, key rotation, safe logging, and graceful fallback |

**Definition of done**: LLM-backed simulations will produce structured citizen reactions and expert summaries without making the app fragile.

### Phase 5 — ML Pipeline

The neural network will learn from LLM-generated examples and scale future simulations.

| Task | Deliverable |
|------|-------------|
| Build feature vector | 40-dimensional Bangladesh-aware feature vector |
| Store training samples | CSV dataset with features and target deltas |
| Train reaction model | MLP model that predicts happiness, support, and income deltas |
| Save model artifacts | Model file and feature scaler saved together |
| Reload model safely | Model and scaler loaded together for inference |
| Show training metrics | Train MAE, validation MAE, sample count, and readiness status |

**Definition of done**: after enough LLM samples are collected, the app will train, save, reload, and use the NN for fast simulations.

### Phase 6 — Analytics Dashboard

The Streamlit interface will make the simulation interactive and explainable.

| Area | Planned Capability |
|------|--------------------|
| Sidebar | Population size, time steps, mode, seed, policy preset, custom policy |
| Learning status | LLM sample count, NN readiness, current mode |
| Overview | Average happiness, support, income, and inequality over time |
| Groups | Breakdown by income, zone, division, religion, and remittance |
| Citizens | Filterable citizen table, detail view, timeline, diary entries |
| Experts | Bangladesh-specific expert analysis |
| NN Analytics | LLM/NN/rule usage, speed, estimated cost savings, training stats |
| Scenarios | Saved runs and side-by-side comparison |

**Definition of done**: a user will configure, run, inspect, train, and compare scenarios from one interface.

### Phase 7 — Verification

The project will be testable without requiring API keys.

| Test Layer | Purpose |
|------------|---------|
| Unit tests | Constants, dataclasses, population generation, features, deltas, stats, ML dataset |
| Rule-based simulation tests | Validate bounds, income behavior, domain coverage, and group stats |
| ML smoke tests | Train, predict, save, load, and apply deltas |
| Import smoke tests | Confirm runtime dependencies are installed |
| Manual UI checks | Confirm Streamlit app launches and renders controls |

**Definition of done**: local tests will validate core behavior, while LLM-specific tests will remain optional and clearly documented.

### Phase 8 — Launch Readiness

Nagorik Genesis will ship with clear setup and deployment paths.

| Area | Deliverable |
|------|-------------|
| Local setup | `pip install -r requirements.txt` and `streamlit run app.py` |
| Environment | `.env.example` with exact runtime variable names |
| Streamlit Cloud | Secrets template and deployment notes |
| Data generation | Batch simulation command for LLM training samples |
| Model training | Offline and UI-based NN training paths |
| Documentation | README, API key guide, simulation guide, testing report |

**Definition of done**: a new user will be able to run a small simulation locally and understand how to collect data, train the NN, and scale up.

---

## 6. Data & Model Strategy

Nagorik Genesis will use a teacher-student learning loop.

```text
LLM reactions -> training samples -> neural network -> fast simulations -> more training data
```

| Data Asset | Purpose |
|------------|---------|
| LLM training samples | High-quality teacher examples for NN training |
| Rule-based training samples | Local baseline data for smoke tests and verification |
| Feature scaler | Ensures NN inference uses the same normalization as training |
| Reaction model | Predicts citizen state deltas at scale |

The first NN milestone will target **500+ LLM samples**. Later training passes will target larger and more diverse datasets across all policy domains.

---

## 7. Testing Strategy

| Gate | Required Result |
|------|-----------------|
| Syntax check | All Python files compile |
| Unit tests | Core data, population, feature, stats, and ML behavior pass |
| Simulation verification | All policy presets remain within valid state bounds |
| ML training check | Model trains, saves, loads, and predicts valid deltas |
| Dependency check | Streamlit, Plotly, Pandas, NumPy, Scikit-learn, Joblib, and dotenv import correctly |
| UI smoke check | App starts and renders the main controls |

Quality will be judged by:

| Signal | Target |
|--------|--------|
| Bounds safety | Happiness, support, and income stay valid |
| Policy coverage | All policy domains are represented |
| Group coverage | Income, zone, division, religion, and remittance analysis works |
| NN correctness | Model and scaler are always paired |
| Runtime practicality | Large NN/rule-based runs remain responsive |
| Explainability | Aggregate trends connect back to citizen and expert narratives |

---

## 8. Launch Plan

### Alpha

The alpha will focus on local simulation and verification.

```bash
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

### Training Data Pass

The first data pass will collect LLM samples across the policy presets.

```bash
python3 batch_simulate.py --pop 50 --steps 2 --presets 8
```

### NN Training Pass

The first NN pass will train once enough LLM samples are available.

```bash
python3 train_nn.py
```

### Beta

The beta will focus on scenario comparison, NN_ONLY scaling, and deployment readiness.

| Requirement | Status Needed |
|-------------|---------------|
| Small LLM simulations | Required |
| NN training from collected samples | Required |
| NN_ONLY large simulation | Required |
| Scenario comparison | Required |
| Streamlit Cloud deployment | Required |
| Clear responsible AI messaging | Required |

---

## 9. Future Direction

Nagorik Genesis will grow into a broader civic simulation platform.

| Expansion | Future Capability |
|-----------|-------------------|
| Policy packs | Healthcare, labor, climate, education, taxation, infrastructure |
| Data export | CSV and JSON exports for scenario analysis |
| Scenario library | Saved policies, reusable populations, benchmark runs |
| Model registry | Track model versions, datasets, validation metrics, and training date |
| Expert review | Add human review and annotation workflows |
| Localization framework | Support future country or regional variants |
| Calibration layer | Tune synthetic distributions against trusted public datasets |

---

## Build Principle

```text
Bangladesh context -> synthetic citizens -> policy simulation -> LLM insight -> neural scale -> explainable dashboard
```

Nagorik Genesis will make policy scenarios easier to explore before decisions move from proposal to public impact.
