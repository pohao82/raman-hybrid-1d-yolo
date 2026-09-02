"""
Unified Training, Retraining & Evaluation Pipeline for DenseDetector.

Supports both:
  1. Standard base training on synthetic Raman spectra with Poisson noise.
  2. Hard-data augmented retraining/fine-tuning using hard-mined error sets (.npz).

Usage:
  # 1. Standard base training:
  python main.py

  # 2. Retraining with hard-mined data:
  python main.py --hard-data hard_mined_v1.npz --oversample 5

  # 3. Custom hyperparameters via CLI:
  python main.py --epochs 150 --lr 1e-3 --batch-size 64

Note:
  CLI flags cover TrainConfig (optimization/run knobs) only. Data/physics
  knobs that live on Config (amp_range, K, dataset_mix, N_INPUT_CHANNELS,
  ...) are structural and still edited in configs.py.
"""

import os
import json
import argparse
from dataclasses import asdict
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from configs import Config, TrainConfig
from src.models_peak_predict import DenseDetector
from src.training import training_loop, build_targets_batch
from src.signal_sample_module import generate_dataset
from src.validation import evaluate_model, build_hard_mined_set


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description="DenseDetector Training & Hard-Mining Pipeline")

    p.add_argument("--hard-data", type=str, default=defaults.hard_data_path,
                    help="Path to hard_mined .npz file. If present on disk, augments training data.")
    p.add_argument("--oversample", type=int, default=defaults.hard_oversample,
                    help=f"Oversampling multiplier for hard cases (default: {defaults.hard_oversample}).")
    p.add_argument("--model-name", type=str, default=defaults.model_name,
                    help="Output name for saved model/config files (default: dense_model).")
    p.add_argument("--epochs", type=int, default=defaults.n_epochs)
    p.add_argument("--batch-size", type=int, default=defaults.nbatch)
    p.add_argument("--lr", type=float, default=defaults.learning_rate)
    p.add_argument("--n-train", type=int, default=defaults.n_train)
    p.add_argument("--n-val", type=int, default=defaults.n_val)
    p.add_argument("--n-test", type=int, default=defaults.n_test)
    p.add_argument("--export-hard-data", type=str, default=defaults.export_hard_data,
                    help="Optional path to export newly mined test failures as .npz.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Small shared helpers (dedupe repeated tensor-conversion / logging code)
# --------------------------------------------------------------------------- #
def to_tensors(X: np.ndarray, targets: Sequence[np.ndarray], device: torch.device):
    """Convert a synthetic (X, targets) pair to device tensors."""
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    targets_t = tuple(torch.as_tensor(t, device=device) for t in targets)
    return X_t, targets_t


def make_loader(X: torch.Tensor, targets: Sequence[torch.Tensor], batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(X, *targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_hard_data(path: str, cfg: Config, device: torch.device):
    """Load a hard-mined .npz set and return (X_tensor, targets_tuple)."""
    hard = np.load(path, allow_pickle=True)
    X_hard = hard["X"]                       # (n_hard, n_points)
    peaks_hard = list(hard["peaks"])         # list of (A, pos, gamma) tuples per spectrum
    targets_hard = build_targets_batch(peaks_hard, cfg, device)
    return to_tensors(X_hard, targets_hard, device)


def augment_with_hard_data(X_train, targets_train, X_hard, targets_hard, oversample: int):
    """Concatenate base training pool with `oversample`x copies of hard cases."""
    X_out = torch.cat([X_train] + [X_hard] * oversample, dim=0)
    targets_out = tuple(
        torch.cat([t_base] + [t_hard] * oversample, dim=0)
        for t_base, t_hard in zip(targets_train, targets_hard)
    )
    return X_out, targets_out


def print_eval_summary(summary: dict, n_test: int) -> None:
    print(f"\n--- Diagnostic Evaluation on {n_test} Test Spectra ---")
    print(f"Test Samples:        {summary['n_samples']}")
    print(f"True Positives (TP): {summary['tp']}")
    print(f"False Negatives(FN): {summary['fn']}")
    print(f"False Positives(FP): {summary['fp']}")
    print(f"Precision:           {summary['precision']:.4f}")
    print(f"Recall:              {summary['recall']:.4f}")
    print(f"F1-Score:            {summary['f1']:.4f}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Load configurations
    cfg = Config()
    cfg_train = TrainConfig(
        model_name=args.model_name,
        nbatch=args.batch_size,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        n_epochs=args.epochs,
        learning_rate=args.lr,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== DenseDetector Training Pipeline ===")
    print(f"Device: {device}")
    print(f"Model Name: {cfg_train.model_name}")
    print(f"Epochs: {cfg_train.n_epochs} | Batch Size: {cfg_train.nbatch} | LR: {cfg_train.learning_rate}")

    # ------------------------------------------------------------------- #
    # 1. Base synthetic data generation + target pre-computation
    # ------------------------------------------------------------------- #
    print(f"\nGenerating {cfg_train.n_train} training and {cfg_train.n_val} validation spectra...")
    X_train_np, peaks_train = generate_dataset(cfg_train.n_train, rng, cfg)
    X_val_np, peaks_val = generate_dataset(cfg_train.n_val, rng, cfg)

    print("Pre-computing geometric targets (presence, offset, amplitude, gamma)...")
    targets_train_np = build_targets_batch(peaks_train, cfg, device)
    targets_val_np = build_targets_batch(peaks_val, cfg, device)

    X_train, targets_train = to_tensors(X_train_np, targets_train_np, device)
    X_val, targets_val = to_tensors(X_val_np, targets_val_np, device)

    # ------------------------------------------------------------------- #
    # 2. Optional hard-data augmentation
    # ------------------------------------------------------------------- #
    if args.hard_data:
        if os.path.exists(args.hard_data):
            print(f"\n[Hard-Data Mining] Ingesting hard dataset: {args.hard_data}")
            X_hard, targets_hard = load_hard_data(args.hard_data, cfg, device)
            print(f"- Loaded {len(X_hard)} hard profiles. Applying oversample factor: {args.oversample}x")

            X_train, targets_train = augment_with_hard_data(
                X_train, targets_train, X_hard, targets_hard, args.oversample
            )
            print(f"- Total training pool after augmentation: {len(X_train)} samples")
        else:
            print(f"\n[Warning] Hard data file '{args.hard_data}' not found. Proceeding with base training.")

    train_loader = make_loader(X_train, targets_train, cfg_train.nbatch, shuffle=True)
    val_loader = make_loader(X_val, targets_val, cfg_train.nbatch * 2, shuffle=False)

    # ------------------------------------------------------------------- #
    # 3. Model instantiation & training loop
    # ------------------------------------------------------------------- #
    print(f"\nInitializing DenseDetector (Input channels: {cfg.N_INPUT_CHANNELS})...")
    model = DenseDetector(n_channels_in=cfg.N_INPUT_CHANNELS, base_layers=cfg.BASE_LAYERS).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=cfg_train.scheduler_factor,
        patience=cfg_train.scheduler_patience,
    )

    print("\n--- Training Begins ---")
    model, best_val_loss, best_state = training_loop(
        n_epochs=cfg_train.n_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        opt=optimizer,
        scheduler=scheduler,
        convergence_window=cfg_train.convergence_window,
        convergence_rel_tol=cfg_train.convergence_rel_tol,
        convergence_min_epochs=cfg_train.convergence_min_epochs,
    )
    model.load_state_dict(best_state)
    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.4f}")

    # ------------------------------------------------------------------- #
    # 4. Save checkpoint & config
    # ------------------------------------------------------------------- #
    model_path = f"{cfg_train.model_name}.pt"
    config_path = f"{cfg_train.model_name}_config.json"

    torch.save(model.state_dict(), model_path)
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"Saved model weights to: {model_path}")
    print(f"Saved configuration to: {config_path}")

    # ------------------------------------------------------------------- #
    # 5. Diagnostic test-set evaluation
    # ------------------------------------------------------------------- #
    X_test_np, peaks_test = generate_dataset(cfg_train.n_test, rng, cfg)
    # evaluate_model expects a CPU tensor/array (it calls .numpy() internally),
    # so this intentionally stays off `device` regardless of where training ran.
    X_test = torch.as_tensor(X_test_np, dtype=torch.float32)
    X_test_raw = X_test[:, 0, :] if X_test.dim() > 2 else X_test

    results = evaluate_model(model, cfg, device, X_test_raw, peaks_test)
    print_eval_summary(results["summary"], cfg_train.n_test)

    if args.export_hard_data:
        X_hard_new, peaks_hard_new = build_hard_mined_set(results["fn_records"], results["fp_records"])
        np.savez(args.export_hard_data, X=X_hard_new, peaks=np.array(peaks_hard_new, dtype=object))
        print(f"Exported {len(X_hard_new)} failure profiles to: {args.export_hard_data}")


if __name__ == "__main__":
    main()
