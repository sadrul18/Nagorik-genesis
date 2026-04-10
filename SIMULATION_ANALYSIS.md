# Nagorik Genesis — Simulation Analysis & Improvement Proposal (Revised)

> Revised after reviewing `MODEL_SPECIFICATION.md` and `train_nn.py` — the actual training pipeline.

---

## 1. The #1 Bug: Missing `target_scaler.inverse_transform()` in Simulation

### What `train_nn.py` actually does

The training pipeline in `train_nn.py` is **well-designed**:

1. Combines 1,592 LLM samples (2× weighted) + 17,500 rule-based samples = **20,684 total**
2. Applies `StandardScaler` to features (`feature_scaler.joblib`)
3. **Applies `StandardScaler` to targets** (`target_scaler.joblib`) — solving the multi-scale problem
4. Trains MLP on scaled targets
5. Evaluates with `inverse_transform` back to original scale
6. Saves all 3 artifacts: `citizen_reaction_mlp.joblib`, `feature_scaler.joblib`, `target_scaler.joblib`

The model was trained on **scaled** targets. Its raw output is in StandardScaler space, not real delta space.

### What `simulation.py` does WRONG

```python
# simulation.py — NN prediction path (line ~395):
deltas = existing_model.predict(X_scaled)[0]     # ← Returns SCALED predictions
new_happiness, new_policy_support, new_income = apply_deltas(prev_state, deltas)
# ❌ No target_scaler.inverse_transform() — using scaled values as real deltas!
```

The `target_scaler` is never loaded or used anywhere in `simulation.py` or `app.py`. The model outputs scaled predictions that are directly applied as real deltas.

### What `MODEL_SPECIFICATION.md` says should happen

```python
# Correct inference pipeline (from spec §8):
deltas_scaled = model.predict(X_scaled)
deltas = target_scaler.inverse_transform(deltas_scaled)  # ← THIS IS MISSING
```

### Proof: Before vs After the Fix

Same simulation: 5,000 pop, 5 steps, Economy domain (Fuel Subsidy Removal).

**BROKEN (current code — no inverse transform):**

| Step | Happiness | Support | Income (৳) |
|------|-----------|---------|------------|
| 0    | 0.5493    | 0.0000  | 49,141     |
| 1    | 0.5581    | -0.0322 | 49,141     |
| 5    | 0.4215    | -0.1689 | **49,141** ← flat |

Income Level (Step 5): Low=-0.20, Middle=-0.13, High=-0.11 (monotonic gradient)
Khulna: **-0.46 support, 0.26 happiness** (extreme outlier)

**FIXED (with `target_scaler.inverse_transform()`):**

| Step | Happiness | Support | Income (৳) |
|------|-----------|---------|------------|
| 0    | 0.5493    | 0.0000  | 49,141     |
| 1    | 0.5637    | +0.0371 | 50,804     |
| 5    | 0.5708    | -0.0164 | **51,763** ← moves realistically |

Income Level (Step 5): Low=-0.05, **Middle=+0.04**, High=-0.03 (non-monotonic — realistic!)
Khulna: **-0.24 support, 0.50 happiness** (still lowest, but no longer extreme)

### What the fix changes

| Metric | Broken | Fixed | Verdict |
|--------|--------|-------|---------|
| Income movement | ৳0 (dead) | +৳2,622 over 5 steps | ✅ Fixed |
| Income level monotonicity | Low < Mid < High always | Middle can be highest | ✅ More realistic |
| Khulna support | -0.46 (extreme) | -0.24 (lower but reasonable) | ⚠️ Better but still biased |
| Happiness stability | Drops to 0.42 | Stable at 0.57 | ✅ No longer collapsing |
| Support range | Huge swings (±0.5 raw) | Contained (±0.05 real) | ✅ Proportional |

---

## 2. Remaining Issues After the Fix

### 2.1 Khulna Is Still the Lowest Division (-0.24 support)

With the inverse transform, Khulna predictions are no longer catastrophic but are still consistently negative across domains:

| Domain              | Khulna delta_s | Dhaka delta_s | Gap    |
|---------------------|---------------|---------------|--------|
| Economy             | -0.298        | -0.082        | -0.216 |
| Education           | -0.036        | +0.311        | -0.347 |
| Social              | -0.186        | -0.134        | -0.052 |
| Digital & Technology| -0.114        | +0.031        | -0.145 |
| Climate & Disaster  | -0.110        | +0.218        | -0.328 |
| Healthcare          | **+0.384**    | +0.295        | +0.089 |

Khulna is negative in 6 out of 7 domains. Only Healthcare is positive. This is a **training data imbalance issue** — only 133 LLM samples for Khulna (8.4% of total) vs 523 for Dhaka (32.8%).

### 2.2 Training Data Coverage Gaps

The 20,684 total samples break down as:
- **17,500 rule-based** — cover all 7 domains evenly (2,500 per domain)
- **1,592 LLM** (×2 weight = 3,184) — heavily skewed:

| Division   | LLM samples | Rule-based samples | Effective LLM influence |
|------------|--------------|-------------------|------------------------|
| Dhaka      | 523 (×2=1046)| ~5,250            | 16.6%                  |
| Khulna     | 133 (×2=266) | ~1,312            | 16.9%                  |
| Barisal    | 105 (×2=210) | ~1,050            | 16.7%                  |

The LLM-to-rule ratio is actually roughly consistent per division. But:
- LLM samples for Khulna Economy: only **42 samples** (×2 = 84) out of ~3,325 total Economy samples
- The LLM for those 42 Khulna samples may have returned genuinely negative reactions (coastal division, economy suffers more)
- **Or** the LLM produced noisy/inconsistent outputs for the small Khulna batch

### 2.3 The Rule-Based Engine Has Its Own Khulna Bias

The rule-based engine (`simulation.py: rule_based_update()`) explicitly penalizes Khulna for Climate & Disaster:

```python
if policy_domain == "Climate & Disaster" and division in ["Barishal", "Khulna"]:
    div_modifier = -0.12   # coastal divisions harder hit
```

This is a hard-coded -12% support decrease for Khulna on Climate policy. Among 17,500 rule-based samples, this bias is baked into the 2,500 Climate samples from Khulna and propagates to the NN.

**However**: The rule-based engine has NO Khulna-specific modifier for Economy, Education, Social, Digital, Infrastructure, or Healthcare. The NN learned a general negative Khulna bias from the combined LLM+rule data, not just from the explicit rule.

### 2.4 Cumulative Drift Is Reduced but Not Eliminated

With proper scaling, per-step deltas are small (±0.03 happiness, ±0.08 support), so 5-step accumulation is much less dramatic. But the autoregressive loop still compounds: a citizen who gets slightly lower happiness at step 1 gets a slightly more negative prediction at step 2, etc.

### 2.5 Revised Root Cause Summary

| # | Issue | Severity | Status After Fix |
|---|-------|----------|-----------------|
| 1 | **Missing `target_scaler.inverse_transform()` in `simulation.py`** | **CRITICAL** | 🔧 Fixable in 1 line |
| 2 | Khulna negative bias in LLM training data (42 Economy samples) | Medium | Needs more data |
| 3 | LLM data imbalance (Dhaka 33% vs Barisal 7%) | Medium | Needs stratified collection |
| 4 | Autoregressive cumulative drift over 5 steps | Low (after fix) | Mitigatable |
| 5 | MLP can't capture complex interactions | Low-Medium | Architecture change helps |

**Root cause #1 explains ~80% of the symptoms.** The model is actually decent — it was just used wrong.

---

## 3. Current Speed Problem

| Mode | Population | Steps | Time | Quality |
|------|-----------|-------|------|---------|
| LLM_ONLY (Ollama local) | 100 | 2 | ~1 hour | Good (nuanced, contextual) |
| LLM_ONLY (Gemini API) | 100 | 2 | ~30 min (rate limited) | Good |
| NN_ONLY (broken) | 5,000 | 5 | ~5 seconds | Poor (wrong scale) |
| NN_ONLY (fixed) | 5,000 | 5 | ~5 seconds | **Acceptable** (minor bias) |
| HYBRID | 5,000 | 5 | ~1 hour+ | Mixed (300 LLM + 4700 NN) |

**After fixing the inverse transform**, NN_ONLY becomes usable. But it still has the Khulna bias and can't generate diary entries or nuanced per-citizen narratives. The LLM produces quality narrative results but can't scale past ~100 citizens without taking hours.

---

## 4. Proposed Improvement Plan

### Tier 1: Immediate Fix (Critical Bug)

**Fix the missing `target_scaler.inverse_transform()` in `simulation.py`.**

This requires:
1. Loading `target_scaler.joblib` alongside the model in `app.py`
2. Passing it to `run_simulation()`
3. Applying `target_scaler.inverse_transform()` after every `model.predict()` call

This single fix transforms NN_ONLY from "broken" to "acceptable" quality.

### Tier 2: Improve Current NN (Quick Wins)

These build on the fixed pipeline:

#### 2a. Collect More Balanced LLM Training Data

The current 1,592 LLM samples are skewed. Run `batch_simulate.py` with stratified populations that ensure equal division coverage:

```bash
# Target: ~50 LLM samples per division per domain = 50 × 8 × 7 = 2,800 new samples
python3 batch_simulate.py --pop 200 --steps 3 --presets 8
```

Priority: ensure Khulna, Barisal, Mymensingh, Sylhet each get 50+ Economy samples.

#### 2b. Output Clamping as Safety Net

After inverse transform, clamp predictions to the observed training range:

```python
delta_h = np.clip(delta_h, -0.30, 0.36)    # observed LLM range
delta_s = np.clip(delta_s, -0.90, 1.00)
delta_i = np.clip(delta_i, -65000, 250000)
```

This prevents any remaining extrapolation from small-sample strata.

#### 2c. Use GradientBoosting for Better Interaction Learning

The MLP struggles with feature interactions (division × domain). Replace with:

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(
    GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1)
)
```

GradientBoosting naturally captures non-linear interactions and works better with the current dataset size (20K samples).

### Tier 3: New Simulation Architecture

After fixing the immediate bug and improving the NN, implement a smarter hybrid pipeline:

#### Architecture: "Stratified Anchor + Calibrated NN"

```
┌─────────────────────────────────────────────────────────────────┐
│                  NAGORIK GENESIS v2 PIPELINE                    │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Phase 1    │───▶│   Phase 2    │───▶│     Phase 3      │   │
│  │ LLM Anchors  │    │ Calibration  │    │  NN Prediction   │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│                                                                 │
│  LLM on 60-80       Compute per-group   Apply NN to all 5000+  │
│  stratified          anchor statistics  with calibration bounds │
│  citizens            (mean, std, range)  and step correction    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Phase 1: Stratified LLM Anchors** (~15-20 min with Ollama)

Instead of random sampling 300 citizens, pick **60-80 citizens** covering all key strata:
- 3 income levels × 8 divisions = 24 minimum cells, 2-3 citizens per cell
- Add edge cases: remittance families, bosti, Hijra
- Each stratum gets at least 1 LLM-quality prediction

60 stratified samples > 300 random samples for coverage. Fewer LLM calls = faster.

**Phase 2: Per-Step Calibration Bounds**

After anchors are computed, derive per-stratum statistics:
```
For each (income_level, division) group:
  anchor_mean = mean of anchor deltas in that group
  anchor_std  = std of anchor deltas in that group
  bounds      = [anchor_mean - 2×std, anchor_mean + 2×std]
```

**Phase 3: Calibrated NN Inference** (~3 sec)

For all non-anchor citizens:
1. NN predicts deltas (with proper inverse transform)
2. Per-stratum clamping: shift/clamp NN prediction toward anchor group statistics
3. After each step, compare NN group means to anchor group means — if they diverge, apply correction shift

This prevents the scenario where the NN's learned bias for Khulna diverges from what the LLM actually says for that specific policy.

**Time estimate:**

| Component | Time |
|-----------|------|
| Phase 1: LLM on 60-80 anchors × 5 steps | ~15 min (Ollama) |
| Phase 2: Compute calibration stats | <1 sec |
| Phase 3: NN predict 5,000 × 5 steps + calibration | ~3 sec |
| **Total** | **~15-16 min** |

vs current:
- LLM_ONLY for 5,000: **~50+ hours** (impractical)
- NN_ONLY (fixed): **~5 seconds** (acceptable but biased)
- **New pipeline: ~15 min (good quality + narratives for anchors)**

### Tier 4: Longer-Term Improvements

| Improvement | Impact | Effort |
|-------------|--------|--------|
| **Add interaction features** (income×domain, division×domain) to feature vector | Explicit signals for cross-effects NN struggles to learn | Low |
| **Separate models per target** (3 GBR instead of 1 MLP) | Each target optimized independently, no scale competition | Medium |
| **Active learning** — call LLM for citizens where NN prediction uncertainty is highest | Focused data collection where it matters most | Medium |
| **Embedding-based policy features** (encode policy description → vector) | Model generalizes to new custom policies without retraining | High |
| **Fine-tuned local LLM** (LoRA on Bangladeshi policy corpus) | Faster + more consistent LLM calls | High |
| **Caching + memoization** — reuse LLM results for similar citizen profiles | Avoid redundant LLM calls across simulations | Low |

---

## 5. Recommended Implementation Order

| Step | Action | Expected Impact |
|------|--------|----------------|
| **1** | **Fix `target_scaler.inverse_transform()` in `simulation.py`** | Fixes 80% of all issues. Income moves, support is realistic, Khulna improves from -0.46 to -0.24 |
| **2** | Add output clamping after inverse transform | Safety net against any remaining extreme predictions |
| **3** | Collect more balanced LLM data (stratified by division) | Addresses Khulna/Barisal underrepresentation |
| **4** | Switch MLP to GradientBoosting in `train_nn.py` | Better interaction learning, better with current data size |
| **5** | Implement stratified anchor sampling in HYBRID mode | Smarter LLM usage: fewer calls, better coverage |
| **6** | Add per-step calibration guardrails | Prevents cumulative drift, anchors NN to LLM group stats |
| **7** | Add interaction features (income×domain, division×domain) | Explicit cross-effect signals |

**Step 1 is a 1-line code fix that should be done immediately.** Steps 2-4 are quick wins. Steps 5-7 are the full v2 pipeline.
