import torch
import torch.nn as nn
import optuna
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from cross_class_helpfulness_weighting.model import MLP
from cross_class_helpfulness_weighting.utils import get_cchw_gradients

def get_data():
    args = get_dataset_args()
    args.num_samples = 4000
    data = make_dataset(args)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    # Validation split
    n_val = 500
    X_val = X_train[-n_val:]
    y_val = y_train[-n_val:]
    X_train = X_train[:-n_val]
    y_train = y_train[:-n_val]

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_one_epoch(model, dl_train, optimizer, mode, beta, device):
    model.train()
    epoch_loss = 0
    for x, y in dl_train:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if mode == 'CCHW':
            weighted_grads = get_cchw_gradients(model, x, y, beta)
            for name, p in model.named_parameters():
                if name in weighted_grads:
                    p.grad = weighted_grads[name]
            # For reporting loss
            with torch.no_grad():
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dl_train)

def evaluate(model, x, y, device):
    model.eval()
    with torch.no_grad():
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc

def objective(trial, mode, X_train, y_train, X_val, y_val, device):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    beta = 0.0
    if mode == 'CCHW':
        beta = trial.suggest_float('beta', -2.0, 2.0)

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    model = MLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    epochs = 10
    for epoch in range(epochs):
        train_one_epoch(model, dl_train, optimizer, mode, beta, device)

    acc = evaluate(model, X_val, y_val, device)
    return acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    sys.stdout.flush()

    X_train, y_train, X_val, y_val, X_test, y_test = get_data()

    results = {}
    best_params = {}
    histories = {}

    modes = ['Baseline', 'CCHW']

    for mode in modes:
        print(f"\n--- Tuning {mode} ---")
        sys.stdout.flush()
        study = optuna.create_study(direction='maximize')
        n_trials = 5 # Reduced trials for both
        study.optimize(lambda t: objective(t, mode, X_train, y_train, X_val, y_val, device), n_trials=n_trials)

        best_params[mode] = study.best_params
        print(f"Best params for {mode}: {best_params[mode]}")
        sys.stdout.flush()

        # Final training with multiple seeds
        print(f"Final training for {mode} (3 seeds)...")
        sys.stdout.flush()
        seed_results = []
        seed_histories = []

        for seed in range(3):
            torch.manual_seed(seed)
            model = MLP().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=best_params[mode]['lr'], weight_decay=1e-4)
            beta = best_params[mode].get('beta', 0.0)

            dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

            history = {'train_loss': [], 'test_acc': []}
            epochs = 20 # Reduced epochs
            for epoch in range(epochs):
                loss = train_one_epoch(model, dl_train, optimizer, mode, beta, device)
                test_acc = evaluate(model, X_test, y_test, device)
                history['train_loss'].append(loss)
                history['test_acc'].append(test_acc)
                print(f"Mode {mode} Seed {seed} - Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Test Acc: {test_acc:.4f}")
                sys.stdout.flush()

            seed_results.append(evaluate(model, X_test, y_test, device))
            seed_histories.append(history)

        results[mode] = {
            'mean': np.mean(seed_results),
            'std': np.std(seed_results),
            'all': seed_results
        }
        histories[mode] = seed_histories

    # Output results
    with open('cross_class_helpfulness_weighting/results.txt', 'w') as f:
        for mode in modes:
            f.write(f"Mode: {mode}\n")
            f.write(f"  Best Params: {best_params[mode]}\n")
            f.write(f"  Final Test Accuracy: {results[mode]['mean']:.4f} +/- {results[mode]['std']:.4f}\n")
            f.write(f"  All seeds: {results[mode]['all']}\n")
            f.write("-" * 20 + "\n")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for mode in modes:
        mean_loss = np.mean([h['train_loss'] for h in histories[mode]], axis=0)
        plt.plot(mean_loss, label=mode)
    plt.title('Training Loss (Mean of 3 seeds)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for mode in modes:
        mean_acc = np.mean([h['test_acc'] for h in histories[mode]], axis=0)
        plt.plot(mean_acc, label=mode)
    plt.title('Test Accuracy (Mean of 3 seeds)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('cross_class_helpfulness_weighting/comparison.png')
    print("\nExperiment complete. Results saved in cross_class_helpfulness_weighting/")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
