import torch
import torch.nn as nn
import optuna
import numpy as np
import matplotlib.pyplot as plt
import os
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from gradient_alignment_sample_weighting.model import MLP
from gradient_alignment_sample_weighting.utils import get_gasw_gradients

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

def train_one_epoch(model, dl_train, optimizer, mode, gamma, device):
    model.train()
    epoch_loss = 0
    for x, y in dl_train:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if mode in ['GASW', 'GDSW']:
            weighted_grads = get_gasw_gradients(model, x, y, gamma, mode=mode)
            for name, p in model.named_parameters():
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
    gamma = 0.0
    if mode in ['GASW', 'GDSW']:
        gamma = trial.suggest_float('gamma', 0.1, 5.0)

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 15
    for epoch in range(epochs):
        train_one_epoch(model, dl_train, optimizer, mode, gamma, device)

    acc = evaluate(model, X_val, y_val, device)
    return acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test = get_data()

    results = {}
    best_params = {}
    histories = {}

    modes = ['Baseline', 'GASW', 'GDSW']

    for mode in modes:
        print(f"\n--- Tuning {mode} ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, mode, X_train, y_train, X_val, y_val, device), n_trials=7)

        best_params[mode] = study.best_params
        print(f"Best params for {mode}: {best_params[mode]}")

        # Final training with best params
        print(f"Final training for {mode}...")
        model = MLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params[mode]['lr'])
        gamma = best_params[mode].get('gamma', 0.0)

        dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

        history = {'train_loss': [], 'test_acc': []}
        epochs = 60
        for epoch in range(epochs):
            loss = train_one_epoch(model, dl_train, optimizer, mode, gamma, device)
            test_acc = evaluate(model, X_test, y_test, device)
            history['train_loss'].append(loss)
            history['test_acc'].append(test_acc)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Test Acc: {test_acc:.4f}")

        results[mode] = evaluate(model, X_test, y_test, device)
        histories[mode] = history

    # Output results
    with open('gradient_alignment_sample_weighting/results.txt', 'w') as f:
        for mode in modes:
            f.write(f"Mode: {mode}\n")
            f.write(f"  Best Params: {best_params[mode]}\n")
            f.write(f"  Final Test Accuracy: {results[mode]:.4f}\n")
            f.write("-" * 20 + "\n")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for mode in modes:
        plt.plot(histories[mode]['train_loss'], label=mode)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for mode in modes:
        plt.plot(histories[mode]['test_acc'], label=mode)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('gradient_alignment_sample_weighting/comparison.png')
    print("\nExperiment complete. Results saved in gradient_alignment_sample_weighting/")

if __name__ == "__main__":
    main()
