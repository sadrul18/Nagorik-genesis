"""
Train the MLP neural network for নাগরিক-GENESIS.
Combines LLM-generated samples + rule-based samples, trains the NN, saves model + scaler.

Usage:
  python3 train_nn.py                     # Train with defaults
  python3 train_nn.py --llm-weight 3.0    # Weight LLM samples 3x vs rule-based
  python3 train_nn.py --epochs 1000       # More training iterations
"""
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

from sklearn.preprocessing import StandardScaler
from ml_data import MLDataset, split_dataset, normalize_features
from nn_model import CitizenReactionModel


def main():
    parser = argparse.ArgumentParser(description="Train নাগরিক-GENESIS MLP model")
    parser.add_argument("--llm-weight", type=float, default=2.0,
                        help="How many times to repeat LLM samples vs rule-based (default: 2.0)")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Max training iterations (default: 500)")
    parser.add_argument("--layers", type=str, default="128,64,32",
                        help="Hidden layer sizes comma-separated (default: 128,64,32)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    hidden_layers = tuple(int(x) for x in args.layers.split(","))

    print("=" * 60)
    print("  নাগরিক-GENESIS — Neural Network Training")
    print("=" * 60)

    # ── Load LLM data ────────────────────────────────────────
    llm_path = os.path.join(_PROJECT_DIR, "data", "llm_training_samples.csv")
    llm_dataset = MLDataset()
    if os.path.exists(llm_path):
        llm_dataset.load_from_csv(llm_path)
        print(f"\n📂 LLM samples:        {llm_dataset.size()}")
    else:
        print(f"\n⚠️  No LLM data found at {llm_path}")

    # ── Load rule-based data ─────────────────────────────────
    rb_path = os.path.join(_PROJECT_DIR, "data", "rule_based_training_data.csv")
    rb_dataset = MLDataset()
    if os.path.exists(rb_path):
        rb_dataset.load_from_csv(rb_path)
        print(f"📂 Rule-based samples: {rb_dataset.size()}")
    else:
        print(f"⚠️  No rule-based data found at {rb_path}")

    if llm_dataset.size() == 0 and rb_dataset.size() == 0:
        print("\n❌ No training data available. Run simulations first.")
        sys.exit(1)

    # ── Combine datasets with LLM weighting ──────────────────
    X_parts = []
    Y_parts = []

    if rb_dataset.size() > 0:
        X_rb, Y_rb = rb_dataset.get_arrays()
        X_parts.append(X_rb)
        Y_parts.append(Y_rb)

    if llm_dataset.size() > 0:
        X_llm, Y_llm = llm_dataset.get_arrays()
        repeats = max(1, int(round(args.llm_weight)))
        for _ in range(repeats):
            X_parts.append(X_llm)
            Y_parts.append(Y_llm)
        print(f"\n🔁 LLM samples repeated {repeats}x (weight={args.llm_weight})")

    X = np.vstack(X_parts)
    Y = np.vstack(Y_parts)
    print(f"📊 Total training samples: {len(X)} ({X.shape[1]} features, {Y.shape[1]} targets)")

    # ── Train/val split ──────────────────────────────────────
    X_train, X_val, Y_train, Y_val = split_dataset(X, Y, args.test_size, args.seed)
    X_train_scaled, X_val_scaled, scaler = normalize_features(X_train, X_val)

    # Scale targets so income (BDT thousands) doesn't dominate happiness/support (0-1)
    y_scaler = StandardScaler()
    Y_train_scaled = y_scaler.fit_transform(Y_train)
    Y_val_scaled = y_scaler.transform(Y_val)

    print(f"\n🔀 Train: {len(X_train)}, Validation: {len(X_val)}")
    print(f"🧠 Architecture: {hidden_layers}")
    print(f"⚙️  Max iterations: {args.epochs}")
    print(f"📏 Target scaling: ON (StandardScaler)")
    print()

    # ── Train ────────────────────────────────────────────────
    model = CitizenReactionModel(
        hidden_layers=hidden_layers,
        max_iter=args.epochs,
        random_state=args.seed
    )
    model.scaler = scaler
    metrics = model.train(X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled)

    # ── Results (in original scale) ──────────────────────────
    Y_train_pred_scaled = model.predict(X_train_scaled)
    Y_val_pred_scaled = model.predict(X_val_scaled)
    Y_train_pred = y_scaler.inverse_transform(Y_train_pred_scaled)
    Y_val_pred = y_scaler.inverse_transform(Y_val_pred_scaled)

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    train_mae = mean_absolute_error(Y_train, Y_train_pred)
    val_mae = mean_absolute_error(Y_val, Y_val_pred)
    val_mse = mean_squared_error(Y_val, Y_val_pred)

    print(f"\n{'='*60}")
    print(f"  TRAINING RESULTS (original scale)")
    print(f"{'='*60}")
    print(f"  Iterations:      {metrics['n_iterations']}")
    print(f"  Train MAE:       {train_mae:.4f}")
    print(f"  Val MAE:         {val_mae:.4f}")
    print(f"  Val MSE:         {val_mse:.4f}")

    # ── Per-target breakdown ─────────────────────────────────
    target_names = ["delta_happiness", "delta_support", "delta_income"]
    print(f"\n  Per-target Val MAE:")
    for i, name in enumerate(target_names):
        mae_i = np.mean(np.abs(Y_val[:, i] - Y_val_pred[:, i]))
        print(f"    {name:20s}: {mae_i:.6f}")

    # ── Save model + scaler + target scaler ──────────────────
    model_path = os.path.join(_PROJECT_DIR, "models", "citizen_reaction_mlp.joblib")
    scaler_path = os.path.join(_PROJECT_DIR, "models", "feature_scaler.joblib")
    y_scaler_path = os.path.join(_PROJECT_DIR, "models", "target_scaler.joblib")

    model.save(model_path, scaler_path)

    import joblib
    joblib.dump(y_scaler, y_scaler_path)

    print(f"\n  💾 Model saved:         {model_path}")
    print(f"  💾 Feature scaler saved: {scaler_path}")
    print(f"  💾 Target scaler saved:  {y_scaler_path}")
    print(f"\n  ✅ Training complete!")


if __name__ == "__main__":
    main()
