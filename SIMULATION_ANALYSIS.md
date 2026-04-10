# Nagorik Genesis — Simulation Analysis & Improvement Status

---

## 1. Resolved Issues

### ✅ 1.1 Missing `target_scaler.inverse_transform()` — FIXED

The MLP was trained on StandardScaler-transformed targets, but `simulation.py` applied raw scaled predictions as real deltas. `app.py` now loads `target_scaler.joblib` and passes it through to `run_simulation()`, which calls `inverse_transform()` after every `model.predict()`.

### ✅ 1.2 HYBRID Mode Rewired — Stratified Anchor + Calibrated NN — IMPLEMENTED

The old HYBRID mode randomly selected 300 citizens for LLM and filled the rest with uncalibrated NN. The new implementation uses a 3-phase pipeline:

```
Phase 1: Stratified LLM Anchors
  → Select ~72 citizens (3 per income_level × division cell = 3 × 24)
  → Process via LLM (or rule-based fallback)

Phase 2: Compute Calibration Bounds
  → Per-stratum (income_level × division) mean/std from anchor results

Phase 3: Calibrated NN Inference
  → NN predicts remaining citizens with inverse_transform
  → Predictions clamped to [anchor_mean ± 2σ] per stratum
```

**Key advantages over old HYBRID:**

| Aspect | Old HYBRID | New Stratified Anchor |
|--------|-----------|----------------------|
| LLM citizens per step | 300 (random) | ~72 (stratified) |
| Coverage | Random — gaps likely in Khulna, Barisal | Guaranteed — every (income × division) cell covered |
| Calibration | None — NN predictions unconstrained | Per-stratum bounds from anchor statistics |
| LLM calls (5 steps, 5000 pop) | 1,500 | ~360 (4× fewer) |
| NN drift control | None | 2σ clamping per group per step |

---

## 2. Open Issues

### 2.1 Khulna Training Data Bias

Khulna has only 133 LLM samples (8.4%) vs 523 for Dhaka (32.8%). The NN shows negative bias for Khulna across 6 of 7 domains:

| Domain              | Khulna delta_s | Dhaka delta_s | Gap    |
|---------------------|---------------|---------------|--------|
| Economy             | -0.298        | -0.082        | -0.216 |
| Education           | -0.036        | +0.311        | -0.347 |
| Healthcare          | **+0.384**    | +0.295        | +0.089 |

**Mitigation:** The new calibration bounds help at runtime — if LLM anchors for Khulna show reasonable values, NN predictions get clamped toward those. But the underlying NN bias remains until more balanced training data is collected.

### 2.2 MLP Interaction Learning Limitations

The MLP struggles with division×domain and income×domain interactions. GradientBoosting would handle these naturally.

---

## 3. Remaining Improvement Plan

| Step | Action | Expected Impact |
|------|--------|----------------|
| **1** | Collect more balanced LLM data (stratified by division) | Addresses Khulna/Barisal underrepresentation |
| **2** | Switch MLP to GradientBoosting in `train_nn.py` | Better interaction learning with current data |
| **3** | Add interaction features (income×domain, division×domain) | Explicit cross-effect signals |
| **4** | Add output clamping after inverse transform (safety net) | Prevents any extreme extrapolation |

### Longer-Term

| Improvement | Impact | Effort |
|-------------|--------|--------|
| Separate models per target (3 GBR instead of 1 MLP) | Each target optimized independently | Medium |
| Active learning — LLM for high-uncertainty citizens | Focused data collection | Medium |
| Embedding-based policy features | Generalize to new policies | High |
| Fine-tuned local LLM (LoRA on BD policy data) | Faster + more consistent LLM calls | High |
| Caching + memoization of LLM results | Avoid redundant calls | Low |
