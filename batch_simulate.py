"""
Batch LLM simulation for training data collection.
Runs multiple policy presets automatically with rate limiting.

Usage:
  python3 batch_simulate.py              # Run all 8 presets, 50 citizens, 2 steps
  python3 batch_simulate.py --pop 30     # Smaller population
  python3 batch_simulate.py --presets 3  # Only first 3 presets
"""
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

from config import get_settings
from data_models import SimulationConfig, PolicyInput
from population import generate_population
from simulation import run_simulation
from llm_client import create_llm_client
from ml_data import MLDataset
from utils import get_policy_presets
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Batch LLM simulation for NN training data")
    parser.add_argument("--pop", type=int, default=100, help="Population size per run (default: 100)")
    parser.add_argument("--steps", type=int, default=2, help="Simulation steps per run (default: 2)")
    parser.add_argument("--presets", type=int, default=8, help="Number of policy presets to run (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 60)
    print("  NAGORIK-GENESIS — Batch LLM Simulation")
    print("=" * 60)

    # Load settings and create client
    settings = get_settings()
    client = create_llm_client(
        backend=settings.llm_backend,
        ollama_model=settings.ollama_model,
        ollama_host=settings.ollama_host,
        gemini_api_key=settings.gemini_api_key,
        backup_keys=settings.backup_api_keys
    )
    presets = get_policy_presets()[:args.presets]

    # Load existing dataset
    dataset = MLDataset()
    data_path = os.path.join(_PROJECT_DIR, "data", "llm_training_samples.csv")
    if os.path.exists(data_path):
        dataset.load_from_csv(data_path)
        print(f"\n📂 Loaded {dataset.size()} existing LLM samples from {data_path}")
    else:
        print(f"\n📂 No existing data found. Starting fresh.")

    total_new = 0
    calls_estimate = args.pop * args.steps * len(presets)
    time_estimate = calls_estimate * 7 / 60  # ~7s per call for Ollama

    print(f"\n🎯 Plan: {len(presets)} presets × {args.pop} citizens × {args.steps} steps")
    print(f"📊 Estimated LLM calls: ~{calls_estimate}")
    print(f"⏱️  Estimated time: ~{time_estimate:.0f} minutes")
    print(f"🤖 Backend: {settings.llm_backend.upper()}")
    print()

    start_time = time.time()

    for i, preset in enumerate(presets):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(presets)}] {preset['title'][:55]}")
        print(f"Domain: {preset['domain']}")
        print(f"{'='*60}")

        # Generate fresh population with different seed per preset
        citizens = generate_population(args.pop, seed=args.seed + i)

        config = SimulationConfig(
            name=f"batch_{preset['domain']}_{i}",
            population_size=args.pop,
            steps=args.steps,
            policy=PolicyInput(
                title=preset["title"],
                description=preset["description"],
                domain=preset["domain"]
            ),
            mode="LLM_ONLY",
            random_seed=args.seed + i
        )

        try:
            results = run_simulation(config, citizens, client)
            new_samples = results["training_dataset"]

            if new_samples.size() > 0:
                dataset.merge(new_samples)
                total_new += new_samples.size()
                print(f"✅ Collected {new_samples.size()} samples (total: {dataset.size()})")
            else:
                print(f"⚠️  No samples collected (likely all fell back to rule-based)")

            stats = results["nn_stats"]
            print(f"   LLM: {stats['total_llm_calls']}, Rule fallback: {stats['total_rule_based']}")

            # Save after each preset (in case we get interrupted)
            dataset.save_to_csv(data_path)
            print(f"   💾 Saved checkpoint ({dataset.size()} total samples)")

        except Exception as e:
            print(f"❌ Error running preset: {e}")

    elapsed = time.time() - start_time

    # Final save
    dataset.save_to_csv(data_path)

    print(f"\n{'='*60}")
    print(f"  BATCH SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  New samples this run: {total_new}")
    print(f"  Total LLM samples:    {dataset.size()}")
    print(f"  Time taken:           {elapsed/60:.1f} minutes")
    print(f"  Saved to:             {data_path}")

    if dataset.size() >= 500:
        print(f"\n  ✅ You have {dataset.size()} samples — enough to train the NN!")
        print(f"     Next step: python3 train_nn.py")
    else:
        remaining = 500 - dataset.size()
        print(f"\n  ⏳ Need {remaining} more samples to reach 500 minimum for NN training.")
        print(f"     Run batch_simulate.py again tomorrow.")

    print()


if __name__ == "__main__":
    main()
