"""
Simulation quality verification for নাগরিক-GENESIS.
Runs rule-based simulations (no API key needed) across all 8 policy presets
and validates that the simulation mechanics produce realistic outputs.

Run: python3 verify_simulation.py
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_models import CitizenState, PolicyInput, DOMAINS
from population import generate_population
from simulation import rule_based_update
from stats import compute_step_stats, compute_time_series_stats, build_stats_dataframe
from utils import format_income, get_policy_presets, build_feature_vector, compute_deltas
from ml_data import MLDataset


def run_rule_based_simulation(citizens, policy_domain, steps=5, seed=42):
    """Run a pure rule-based simulation and return all states + stats."""
    rng = np.random.default_rng(seed)
    all_states = []

    # Step 0: Initialize
    for c in citizens:
        all_states.append(CitizenState(
            citizen_id=c.id, step=0,
            happiness=c.base_happiness,
            policy_support=0.0,
            income=c.base_income
        ))

    # Steps 1-N
    for step in range(1, steps + 1):
        for c in citizens:
            prev = next(s for s in reversed(all_states) if s.citizen_id == c.id and s.step == step - 1)
            h, s, i = rule_based_update(c, prev, policy_domain, rng)
            all_states.append(CitizenState(
                citizen_id=c.id, step=step,
                happiness=h, policy_support=s, income=i
            ))

    step_stats = compute_time_series_stats(all_states, citizens, steps)
    return all_states, step_stats


def verify_single_policy(citizens, preset, steps=5):
    """Verify simulation for a single policy and return results dict."""
    all_states, step_stats = run_rule_based_simulation(
        citizens, preset["domain"], steps=steps
    )

    results = {
        "policy": preset["title"][:60],
        "domain": preset["domain"],
        "steps": steps,
        "issues": []
    }

    # Check 1: Happiness should stay in [0, 1]
    for s in all_states:
        if not (0.0 <= s.happiness <= 1.0):
            results["issues"].append(f"Happiness {s.happiness} out of bounds")
            break

    # Check 2: Support should stay in [-1, 1]
    for s in all_states:
        if not (-1.0 <= s.policy_support <= 1.0):
            results["issues"].append(f"Support {s.policy_support} out of bounds")
            break

    # Check 3: Income should never be negative
    for s in all_states:
        if s.income < 0:
            results["issues"].append(f"Income {s.income} < 0")
            break

    # Check 4: Stats should have by_division and by_religion
    final_stats = step_stats[-1]
    if not final_stats.by_division:
        results["issues"].append("by_division is empty")
    if not final_stats.by_religion:
        results["issues"].append("by_religion is empty")

    # Check 5: Inequality gap should be reasonable (-0.5 to 0.5)
    gap = final_stats.inequality_gap_happiness
    if abs(gap) > 0.5:
        results["issues"].append(f"Inequality gap {gap:.3f} seems too large")

    # Check 6: Average income should remain positive and realistic
    for stats in step_stats:
        if stats.avg_income <= 0:
            results["issues"].append(f"Step {stats.step}: avg_income <= 0")
            break

    # Check 7: There should be variation across divisions
    if final_stats.by_division and len(final_stats.by_division) >= 3:
        div_happs = [d["avg_happiness"] for d in final_stats.by_division]
        div_range = max(div_happs) - min(div_happs)
        if div_range < 0.001:
            results["issues"].append("No variation across divisions (suspicious)")

    # Step metrics
    step0 = step_stats[0]
    step_final = step_stats[-1]
    results["initial_happiness"] = f"{step0.avg_happiness:.3f}"
    results["final_happiness"] = f"{step_final.avg_happiness:.3f}"
    results["initial_income"] = format_income(step0.avg_income)
    results["final_income"] = format_income(step_final.avg_income)
    results["final_support"] = f"{step_final.avg_support:+.3f}"
    results["happiness_delta"] = f"{step_final.avg_happiness - step0.avg_happiness:+.3f}"
    results["inequality_gap"] = f"{gap:+.3f}"
    results["divisions_tracked"] = len(final_stats.by_division)
    results["religions_tracked"] = len(final_stats.by_religion)
    results["passed"] = len(results["issues"]) == 0

    return results, all_states


def generate_training_data(citizens, all_states_by_policy, presets):
    """Generate training dataset from rule-based simulations."""
    dataset = MLDataset()

    for preset, all_states in zip(presets, all_states_by_policy):
        citizen_map = {c.id: c for c in citizens}
        states_by_citizen_step = {}
        for s in all_states:
            states_by_citizen_step[(s.citizen_id, s.step)] = s

        for s in all_states:
            if s.step == 0:
                continue
            citizen = citizen_map[s.citizen_id]
            prev = states_by_citizen_step[(s.citizen_id, s.step - 1)]

            X = build_feature_vector(citizen, prev, preset["domain"])
            Y = compute_deltas(prev, s.happiness, s.policy_support, s.income)
            dataset.add_sample(X, Y)

    return dataset


def main():
    print("=" * 80)
    print("  নাগরিক-GENESIS — Simulation Quality Verification")
    print("=" * 80)

    # Generate population
    POP_SIZE = 500
    STEPS = 5
    print(f"\n📊 Population: {POP_SIZE} citizens, {STEPS} steps per simulation")
    citizens = generate_population(POP_SIZE, seed=42)

    # Population summary
    income_counts = {"low": 0, "middle": 0, "high": 0}
    for c in citizens:
        income_counts[c.income_level] += 1
    avg_age = np.mean([c.age for c in citizens])
    avg_income = np.mean([c.base_income for c in citizens])
    remit_pct = sum(1 for c in citizens if c.is_remittance_family) / len(citizens) * 100

    print(f"   Income split: Low {income_counts['low']}, Middle {income_counts['middle']}, High {income_counts['high']}")
    print(f"   Avg age: {avg_age:.1f}, Avg income: {format_income(avg_income)}, Remittance: {remit_pct:.1f}%")

    # Run all 8 presets
    presets = get_policy_presets()
    all_results = []
    all_states_list = []

    print(f"\n🔄 Running {len(presets)} policy simulations...\n")
    for i, preset in enumerate(presets):
        result, states = verify_single_policy(citizens, preset, steps=STEPS)
        all_results.append(result)
        all_states_list.append(states)

        status = "✅" if result["passed"] else "❌"
        print(f"  {status} [{i+1}/{len(presets)}] {result['domain']:22s} | "
              f"H: {result['initial_happiness']} → {result['final_happiness']} "
              f"({result['happiness_delta']}) | "
              f"Income: {result['initial_income']} → {result['final_income']} | "
              f"Support: {result['final_support']} | "
              f"Gap: {result['inequality_gap']}")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"     ⚠️  {issue}")

    # Summary
    passed = sum(1 for r in all_results if r["passed"])
    print(f"\n{'=' * 80}")
    print(f"  Verification: {passed}/{len(all_results)} policies passed all checks")
    print(f"{'=' * 80}")

    # Check for domain diversity
    print("\n📋 Domain Diversity Check:")
    domains_tested = set(r["domain"] for r in all_results)
    for d in DOMAINS:
        status = "✅" if d in domains_tested else "❌"
        print(f"  {status} {d}")

    # Generate training data
    print(f"\n💾 Generating training data from rule-based simulations...")
    dataset = generate_training_data(citizens, all_states_list, presets)
    X, Y = dataset.get_arrays()

    print(f"   Samples generated: {dataset.size()}")
    print(f"   Feature dimensions: {X.shape[1]}")
    print(f"   Target dimensions: {Y.shape[1]}")

    # Save training data
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rule_based_training_data.csv")
    dataset.save_to_csv(output_path)
    print(f"   Saved to: {output_path}")

    # Data quality stats
    print(f"\n📊 Training Data Quality:")
    print(f"   Delta Happiness — mean: {np.mean(Y[:, 0]):.4f}, std: {np.std(Y[:, 0]):.4f}, "
          f"range: [{np.min(Y[:, 0]):.4f}, {np.max(Y[:, 0]):.4f}]")
    print(f"   Delta Support   — mean: {np.mean(Y[:, 1]):.4f}, std: {np.std(Y[:, 1]):.4f}, "
          f"range: [{np.min(Y[:, 1]):.4f}, {np.max(Y[:, 1]):.4f}]")
    print(f"   Delta Income    — mean: {np.mean(Y[:, 2]):.1f}, std: {np.std(Y[:, 2]):.1f}, "
          f"range: [{np.min(Y[:, 2]):.0f}, {np.max(Y[:, 2]):.0f}]")

    # Quick NN training on rule-based data
    print(f"\n🧠 Quick NN training on rule-based data...")
    from nn_model import CitizenReactionModel
    from ml_data import split_dataset, normalize_features

    X_train, X_val, Y_train, Y_val = split_dataset(X, Y, test_size=0.2, random_state=42)
    X_train_s, X_val_s, scaler = normalize_features(X_train, X_val)

    model = CitizenReactionModel(hidden_layers=(128, 64, 32), max_iter=300)
    model.scaler = scaler
    metrics = model.train(X_train_s, Y_train, X_val_s, Y_val)

    print(f"   Train MAE: {metrics['train_mae']:.4f}")
    print(f"   Val MAE:   {metrics.get('val_mae', 'N/A')}")
    print(f"   Iterations: {metrics['n_iterations']}")

    # Save model
    model.save("models/citizen_reaction_mlp.joblib", "models/feature_scaler.joblib")
    print(f"   Model saved to models/")

    print(f"\n✅ Done! {dataset.size()} training samples generated and NN trained.")
    print(f"   You can now run the Streamlit app in HYBRID or NN_ONLY mode.")

    return 0 if passed == len(all_results) else 1


if __name__ == "__main__":
    exit(main())
