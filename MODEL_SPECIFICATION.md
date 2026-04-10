# নাগরিক-GENESIS — Model Specification & ML Pipeline

## Table of Contents

1. [Overview](#1-overview)
2. [Data Generation Pipeline](#2-data-generation-pipeline)
   - 2.1 [Synthetic Population](#21-synthetic-population)
   - 2.2 [Policy Presets](#22-policy-presets)
   - 2.3 [LLM Data Generation](#23-llm-data-generation)
   - 2.4 [Rule-Based Data Generation](#24-rule-based-data-generation)
3. [Feature Engineering](#3-feature-engineering)
4. [Target Variables](#4-target-variables)
5. [Data Cleaning](#5-data-cleaning)
6. [Model Architecture](#6-model-architecture)
7. [Training Pipeline](#7-training-pipeline)
8. [Inference Pipeline](#8-inference-pipeline)
9. [Training Results](#9-training-results)
10. [File Reference](#10-file-reference)

---

## 1. Overview

নাগরিক-GENESIS (Nagorik-GENESIS) is a Bangladesh-localized citizen reaction simulation system. It predicts how synthetic Bangladeshi citizens respond to government policy changes across 3 dimensions: **happiness**, **policy support**, and **income**.

The core idea: use an LLM (Ollama/qwen2.5:7b or Google Gemini) to generate high-quality citizen reaction data, then **distill** that knowledge into a lightweight MLP neural network that can run predictions 1000x faster without needing an LLM at inference time.

**Pipeline summary:**

```
┌──────────────────────┐     ┌──────────────────────┐
│  Synthetic Population │────▶│   LLM Simulation     │
│  (100 citizens × 8   │     │   (Ollama qwen2.5:7b)│
│   policy presets)     │     │   ~7s per citizen     │
└──────────────────────┘     └────────┬─────────────┘
                                      │
                                      ▼
                              ┌───────────────┐       ┌──────────────────────┐
                              │  1,592 LLM    │       │  17,500 Rule-Based   │
                              │  samples      │       │  samples             │
                              └───────┬───────┘       └──────────┬───────────┘
                                      │                          │
                                      ▼                          ▼
                              ┌─────────────────────────────────────┐
                              │   Combined Dataset (20,684 rows)    │
                              │   LLM repeated 2× for weighting     │
                              └──────────────┬──────────────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────────┐
                              │   MLP Neural Network Training   │
                              │   Input: 40D → (128,64,32) → 3D│
                              └──────────────┬──────────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────────┐
                              │   Saved Artifacts:              │
                              │   • citizen_reaction_mlp.joblib │
                              │   • feature_scaler.joblib       │
                              │   • target_scaler.joblib        │
                              └─────────────────────────────────┘
```

---

## 2. Data Generation Pipeline

### 2.1 Synthetic Population

Each simulation generates a population of synthetic Bangladeshi citizens using `population.py`. Every citizen has **15 attributes** drawn from real Bangladesh demographics:

| Attribute             | Type        | Distribution / Range                                                                 |
|-----------------------|-------------|--------------------------------------------------------------------------------------|
| `age`                 | int         | Weighted toward youth (median ~27), range 18–80                                      |
| `gender`              | categorical | 50.5% Male, 49.2% Female, 0.3% Hijra                                                |
| `income_level`        | categorical | 55% low (৳8–15K/mo), 35% middle (৳20–60K/mo), 10% high (৳80K–5L/mo)                |
| `city_zone`           | categorical | shohor_kendro, shilpo_elaka, uposhohon, graam, bosti (5 zones)                       |
| `division`            | categorical | 8 divisions, population-weighted (Dhaka 30%, Chittagong 18%, etc.)                   |
| `religion`            | categorical | 91% Muslim, 8% Hindu, 1% other                                                      |
| `education`           | categorical | Level tied to income bracket                                                         |
| `profession`          | categorical | Tied to income level (e.g., garment worker → low, software engineer → high)          |
| `political_view`      | categorical | Government Supporter, Opposition, Neutral, Progressive, Islamist (4 categories)      |
| `family_size`         | int         | Range 1–10, avg ~4.5                                                                 |
| `risk_tolerance`      | float       | 0.0–1.0, personality trait                                                           |
| `openness_to_change`  | float       | 0.0–1.0, personality trait                                                           |
| `base_happiness`      | float       | 0.0–1.0, initial happiness                                                           |
| `base_income`         | float       | Monthly income in BDT (৳), drawn from income_level bracket                           |
| `is_remittance_family`| bool        | ~12% of families receive overseas remittance                                          |

### 2.2 Policy Presets

Training data is collected across **8 preset Bangladeshi policies** covering all 7 domains:

| # | Policy (Bengali + English)                                   | Domain               |
|---|--------------------------------------------------------------|----------------------|
| 1 | জ্বালানি ভর্তুকি প্রত্যাহার (Fuel Subsidy Removal)           | Economy              |
| 2 | গার্মেন্ট শ্রমিকদের মজুরি বৃদ্ধি (RMG Minimum Wage Hike)     | Economy              |
| 3 | ডিজিটাল বাংলাদেশ বৃত্তি (Digital Bangladesh Scholarship)     | Education            |
| 4 | বন্যা-পরবর্তী সহায়তা (Post-Flood Relief Package)            | Climate & Disaster   |
| 5 | মেট্রোরেল সম্প্রসারণ (Metro Rail Expansion)                  | Infrastructure       |
| 6 | বস্তি উচ্ছেদ ও পুনর্বাসন (Slum Eviction & Resettlement)      | Social               |
| 7 | সকলের জন্য স্বাস্থ্যসেবা কার্ড (Universal Health Card)       | Healthcare           |
| 8 | ফ্রিল্যান্সিং ট্যাক্স প্রণোদনা (Freelancing Tax Incentive)   | Digital & Technology |

Each preset runs with **100 citizens × 2 simulation steps = 200 LLM calls** per preset.

### 2.3 LLM Data Generation

**Backend:** Ollama running `qwen2.5:7b` (4.7GB, Q4_K_M quantization) on an NVIDIA RTX 3060 Laptop GPU (6GB VRAM).

**Process (`batch_simulate.py` → `simulation.py` → `llm_client.py`):**

1. For each citizen at each step, `simulation.py` constructs a **40-dimensional feature vector** (see §3).
2. The citizen's profile, current state, and policy are sent to the LLM as a structured prompt.
3. The LLM returns a **JSON response** with 5 fields.
4. Deltas (changes) are computed and stored as training targets.

**LLM System Prompt (abridged):**

The system prompt provides deep Bangladesh context to ground the LLM's responses:

```
You are simulating how a fictional Bangladeshi citizen (নাগরিক) reacts to a new 
government policy or event. You receive citizen_profile, current_state, and policy.

CRITICAL CONTEXT — BANGLADESH (বাংলাদেশ):
- Currency: BDT (৳). ৳1 USD ≈ ৳110 BDT.
- Income: ৳8-15K/month = low, ৳20-60K = middle, ৳80K+ = upper
- Key sectors: RMG (4M+ workers, 80% women), Agriculture (40%), Remittance (6% GDP)
- Geography: Low-lying delta, vulnerable to floods/cyclones
- Urban: Dhaka 47,000 people/km², bosti ~40% of urban pop
- Rural: 63% of population, agriculture + remittance based
- Digital: 130M+ internet users, bKash/Nagad widely used

Output ONLY valid JSON:
- new_happiness: float [0, 1]
- new_policy_support: float [-1, 1]
- income_delta: float in BDT (monthly change)
- short_reason: 1-2 sentence explanation
- diary_entry: 3-5 sentences, first-person as a Bangladeshi citizen
```

**User prompt format:**

```json
{
  "citizen_profile": { "age": 34, "gender": "Female", "income_level": "low", ... },
  "current_state": { "happiness": 0.45, "policy_support": 0.1, "income": 12000 },
  "policy": { "title": "Fuel Subsidy Removal", "description": "...", "domain": "Economy" }
}
```

**LLM response → Training sample conversion:**

```python
# From llm_client.py: parse JSON response
reaction = {
    "new_happiness":      clamp(response["new_happiness"], 0.0, 1.0),
    "new_policy_support": clamp(response["new_policy_support"], -1.0, 1.0),
    "income_delta":       float(response["income_delta"]),
}

# From simulation.py: compute deltas as training targets
delta_happiness = new_happiness - prev_state.happiness
delta_support   = new_policy_support - prev_state.policy_support
delta_income    = new_income - prev_state.income

# Store: X = feature_vector (40D), Y = [delta_h, delta_s, delta_i]
training_dataset.add_sample(X, Y)
```

**Timing:** ~7 seconds per LLM call (warm), ~42s cold start. Total for 8 presets: **252.6 minutes** (4.2 hours).

### 2.4 Rule-Based Data Generation

A separate **7-factor rule-based engine** (`simulation.py: rule_based_update()`) generates 17,500 additional samples without needing an LLM. This data is noisier but provides broad coverage.

**The 7 factors:**

| # | Factor                         | Description                                                          |
|---|--------------------------------|----------------------------------------------------------------------|
| 1 | Domain × Income Level          | Base income delta: domain-specific ranges per income bracket         |
| 2 | Zone Multiplier                | bosti=1.6×, urban_poor=1.3×, urban_rich=0.6× (vulnerability scaling)|
| 3 | Profession × Domain Affinity   | Garment workers boost on Economy, Farmers on Climate, etc.           |
| 4 | Division × Domain Effects      | Coastal divisions (Barishal, Khulna) hit harder by Climate; Dhaka benefits from Infrastructure |
| 5 | Remittance Family Boost        | +6–14% happiness for remittance families under Economy policies      |
| 6 | Family Size Amplifier          | Large families (≥5) get +3% per extra member on Social/Healthcare    |
| 7 | State Momentum (Despair Amp.)  | Very unhappy citizens (happiness < 0.3) react more dramatically      |

**Plus:** Personality modifiers (openness, risk tolerance), political view support bias (Government Supporter +8–18%, Opposition −6–18%), and Gaussian noise (±0.05).

---

## 3. Feature Engineering

Each citizen×step×policy combination is encoded into a **40-dimensional feature vector** by `utils.py: build_feature_vector()`:

| Dimensions | Feature                    | Encoding               | Range          |
|------------|----------------------------|------------------------|----------------|
| 1          | Age                        | Normalized: age/100    | [0.18, 0.80]   |
| 3          | Income Level               | One-hot: low/mid/high  | {0, 1}         |
| 5          | City Zone                  | One-hot: 5 zones       | {0, 1}         |
| 4          | Political View             | One-hot: 4 views       | {0, 1}         |
| 1          | Risk Tolerance             | Raw float              | [0.0, 1.0]     |
| 1          | Openness to Change         | Raw float              | [0.0, 1.0]     |
| 1          | Family Size                | Normalized: size/10    | [0.1, 1.0]     |
| 1          | Previous Happiness         | Raw float              | [0.0, 1.0]     |
| 1          | Previous Policy Support    | Raw float              | [−1.0, 1.0]    |
| 1          | Previous Income            | Log-scaled: log1p/15   | [0.0, ~1.0]    |
| 7          | Policy Domain              | One-hot: 7 domains     | {0, 1}         |
| 8          | Division                   | One-hot: 8 divisions   | {0, 1}         |
| 5          | Religion                   | One-hot: 5 categories  | {0, 1}         |
| 1          | Is Remittance Family       | Binary                 | {0, 1}         |
| **40**     | **Total**                  |                        |                |

**Encoding details:**
- **One-hot categorical** features use `encode_categorical_one_hot()` — if the value isn't in the category list, all positions are 0.
- **Income** is log-scaled (`log1p(income) / 15.0`) to handle the wide BDT range (৳8K–৳500K) and compress it into a ~[0, 1] range.
- **Age** and **family size** are linearly normalized to [0, 1].

---

## 4. Target Variables

The model predicts **3 continuous targets** (deltas from previous state):

| Target              | Description                                    | Typical Range        | Unit      |
|---------------------|------------------------------------------------|----------------------|-----------|
| `delta_happiness`   | Change in happiness score                      | [−0.28, +0.36]       | unitless  |
| `delta_support`     | Change in policy support score                 | [−0.90, +1.00]       | unitless  |
| `delta_income`      | Change in monthly income                       | [−65K, +250K]        | BDT (৳)   |

At inference time, deltas are applied to the previous state and clamped:
- `new_happiness = clip(prev_happiness + delta_h, 0.0, 1.0)`
- `new_support = clip(prev_support + delta_s, -1.0, 1.0)`
- `new_income = max(0.0, prev_income + delta_i)`

---

## 5. Data Cleaning

Before training, the raw 1,600 LLM-generated samples are cleaned:

| Issue                          | Count   | Action                      |
|--------------------------------|---------|-----------------------------|
| All-zero targets (garbage)     | 5 rows  | Removed                     |
| Extreme income (>\|500K\| BDT) | 3 rows  | Removed                     |
| NaN values                     | 0 rows  | —                           |
| Inf values                     | 0 rows  | —                           |
| **Final clean LLM samples**   | **1,592** |                           |

**Post-cleaning delta statistics:**

| Target            | Mean       | Std        | Min        | Max         |
|-------------------|------------|------------|------------|-------------|
| delta_happiness   | 0.0081     | 0.0602     | −0.2800    | 0.3588      |
| delta_support     | 0.1354     | 0.4640     | −0.9000    | 1.0000      |
| delta_income      | 3,135 ৳    | 22,307 ৳   | −64,770 ৳  | 250,000 ৳   |

---

## 6. Model Architecture

**Type:** Multi-Layer Perceptron (MLP) Regressor from scikit-learn

```
Input Layer:     40 neurons  (40-D feature vector)
                     │
                     ▼
Hidden Layer 1:  128 neurons  (ReLU activation)
                     │
                     ▼
Hidden Layer 2:   64 neurons  (ReLU activation)
                     │
                     ▼
Hidden Layer 3:   32 neurons  (ReLU activation)
                     │
                     ▼
Output Layer:      3 neurons  (delta_happiness, delta_support, delta_income)
```

**Hyperparameters:**

| Parameter              | Value              | Rationale                                       |
|------------------------|--------------------|-------------------------------------------------|
| `hidden_layer_sizes`   | (128, 64, 32)      | Decreasing width for 40-dim input               |
| `activation`           | ReLU               | Standard for regression tasks                   |
| `max_iter`             | 500                | Upper bound (early stopping usually triggers)   |
| `early_stopping`       | True               | Prevents overfitting                            |
| `validation_fraction`  | 0.1                | Internal validation split for early stopping    |
| `n_iter_no_change`     | 10                 | Patience: stop after 10 iterations no improvement|
| `random_state`         | 42                 | Reproducibility                                 |

**Total parameters:** ~14,435
- Layer 1: 40×128 + 128 = 5,248
- Layer 2: 128×64 + 64 = 8,256
- Layer 3: 64×32 + 32 = 2,080
- Output: 32×3 + 3 = 99
- Biases accounted for above

---

## 7. Training Pipeline

**Script:** `train_nn.py`

### Step-by-step:

```
1. LOAD DATA
   ├── data/llm_training_samples.csv      → 1,592 rows × 43 cols
   └── data/rule_based_training_data.csv  → 17,500 rows × 43 cols

2. COMBINE WITH WEIGHTING
   ├── Rule-based: 17,500 (1×)
   └── LLM: 1,592 × 2 repeats = 3,184 (2× weight)
   └── Total: 20,684 samples

3. TRAIN/VAL SPLIT
   ├── Train: 16,547 (80%)
   └── Val:    4,137 (20%)
   └── Stratified random split, seed=42

4. FEATURE SCALING (StandardScaler)
   ├── Fit on X_train only
   └── Transform both X_train and X_val
   └── Saves: feature_scaler.joblib

5. TARGET SCALING (StandardScaler)
   ├── Fit on Y_train only
   └── Transform both Y_train and Y_val
   └── Purpose: Prevent delta_income (thousands BDT) from
   │   dominating delta_happiness/support (0–1 scale)
   └── Saves: target_scaler.joblib

6. TRAIN MLP
   ├── MLPRegressor with early stopping
   ├── Optimizer: Adam (scikit-learn default)
   └── Converged at 66 iterations

7. EVALUATE (inverse-transform predictions to original scale)
   ├── Per-target Val MAE computed
   └── Overall MAE computed

8. SAVE ARTIFACTS
   ├── models/citizen_reaction_mlp.joblib
   ├── models/feature_scaler.joblib
   └── models/target_scaler.joblib
```

### Why LLM samples are weighted 2×

LLM-generated data is higher quality (contextually grounded in Bangladesh realities via the system prompt) but expensive to produce (~7s/call). Rule-based data provides volume but uses simpler heuristics. Repeating LLM samples 2× in the training set gives them proportionally more influence:

- Without weighting: 1,592 / 19,092 = **8.3%** LLM influence
- With 2× weighting: 3,184 / 20,684 = **15.4%** LLM influence

### Why target scaling is critical

Without target scaling, `delta_income` (range: −65K to +250K BDT) has ~100,000× larger magnitude than `delta_happiness` (range: −0.28 to +0.36). The MLP's loss function (MSE) would optimize almost entirely for income, causing early stopping at iteration 12 with poor happiness/support predictions.

| Metric                | Without Target Scaling | With Target Scaling |
|-----------------------|------------------------|---------------------|
| Iterations            | 12                     | **66**              |
| delta_happiness MAE   | 0.2848                 | **0.0491**          |
| delta_support MAE     | 0.2875                 | **0.0997**          |
| delta_income MAE      | 1,937 ৳                | **1,775 ৳**         |

---

## 8. Inference Pipeline

At simulation time, the trained model is used in two possible modes:

### Mode 1: NN_ONLY (fast, ~0.001s per citizen)

```python
# 1. Build feature vector
X = build_feature_vector(citizen, prev_state, policy_domain)  # 40-D

# 2. Scale features
X_scaled = feature_scaler.transform(X.reshape(1, -1))

# 3. Predict (scaled targets)
deltas_scaled = model.predict(X_scaled)  # [delta_h', delta_s', delta_i']

# 4. Inverse-transform targets back to original scale
deltas = target_scaler.inverse_transform(deltas_scaled)  # [delta_h, delta_s, delta_i]

# 5. Apply deltas with clamping
new_happiness = clip(prev_happiness + delta_h, 0.0, 1.0)
new_support   = clip(prev_support + delta_s, -1.0, 1.0)
new_income    = max(0.0, prev_income + delta_i)
```

### Mode 2: HYBRID (balanced quality + speed)

```
For each citizen in population:
├── If citizen selected for LLM sample (~300 per step):
│   └── Call LLM → get reaction → store as training data
├── Else if trained NN model is available:
│   └── NN prediction (fast)
└── Else:
    └── Rule-based fallback (7-factor engine)
```

### Mode 3: LLM_ONLY (highest quality, slowest)

All citizens are sent to the LLM. Used exclusively for **training data collection**, not production.

### Fallback chain

```
LLM call attempted
  ├── Success → use LLM result, store as training sample
  └── Failure (timeout, parse error, quota) →
       ├── NN available → NN prediction
       └── NN unavailable → Rule-based (7-factor engine)
```

---

## 9. Training Results

### Final model performance

| Metric                  | Value           |
|-------------------------|-----------------|
| Total training samples  | 20,684          |
| Train split             | 16,547 (80%)    |
| Validation split        | 4,137 (20%)     |
| Iterations to converge  | 66              |
| Architecture            | (128, 64, 32)   |
| Overall Train MAE       | 545.25          |
| Overall Val MAE         | 591.59          |

### Per-target validation MAE

| Target            | Val MAE   | Interpretation                            |
|-------------------|-----------|-------------------------------------------|
| delta_happiness   | 0.0491    | ±0.05 on a 0–1 scale → excellent          |
| delta_support     | 0.0997    | ±0.10 on a −1 to 1 scale → good           |
| delta_income      | 1,775 ৳   | ±1,775 BDT/month → acceptable for BD context |

### Data composition

| Source        | Raw Samples | After Cleaning | In Training (with weight) |
|---------------|-------------|----------------|---------------------------|
| LLM (Ollama)  | 1,600       | 1,592          | 3,184 (2×)                |
| Rule-based    | 17,500      | 17,500         | 17,500 (1×)               |
| **Total**     | **19,100**  | **19,092**     | **20,684**                |

---

## 10. File Reference

| File                    | Role                                                      |
|-------------------------|-----------------------------------------------------------|
| `train_nn.py`           | Training script — loads data, trains MLP, saves artifacts |
| `nn_model.py`           | `CitizenReactionModel` class (MLP wrapper)                |
| `ml_data.py`            | `MLDataset` class, split/normalize utilities              |
| `simulation.py`         | Simulation engine — orchestrates LLM/NN/rule-based        |
| `llm_client.py`         | `OllamaClient` + `GeminiClient` — LLM backends           |
| `batch_simulate.py`     | Batch data collection across 8 policy presets             |
| `population.py`         | Synthetic Bangladeshi population generator                |
| `data_models.py`        | Dataclasses: Citizen, CitizenState, PolicyInput, etc.     |
| `utils.py`              | Feature vector builder, encoding, deltas, presets         |
| `config.py`             | Settings from .env (backend, model, keys)                 |

### Saved model artifacts

| File                              | Contents                                      |
|-----------------------------------|-----------------------------------------------|
| `models/citizen_reaction_mlp.joblib` | Trained MLPRegressor (scikit-learn)          |
| `models/feature_scaler.joblib`       | StandardScaler fitted on X_train (40-D)     |
| `models/target_scaler.joblib`        | StandardScaler fitted on Y_train (3-D)      |

### Training data files

| File                                | Rows    | Columns | Source            |
|-------------------------------------|---------|---------|-------------------|
| `data/llm_training_samples.csv`     | 1,592   | 43      | Ollama qwen2.5:7b |
| `data/rule_based_training_data.csv` | 17,500  | 43      | 7-factor engine   |

Column format: `feature_0` through `feature_39` (40 features) + `delta_happiness`, `delta_support`, `delta_income` (3 targets).
