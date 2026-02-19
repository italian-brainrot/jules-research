import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import numpy as np
import matplotlib.pyplot as plt
import os
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from pcgd_experiment.optimizer import PCGDOptimizer

class MLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_data(noisy_fraction=0.0):
    args = get_dataset_args()
    args.num_samples = 4000
    data = make_dataset(args)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    if noisy_fraction > 0:
        n_noisy = int(noisy_fraction * len(y_train))
        noisy_indices = torch.randperm(len(y_train))[:n_noisy]
        y_train[noisy_indices] = torch.randint(0, 10, (n_noisy,))

    # Validation split
    n_val = 500
    X_val = X_train[-n_val:]
    y_val = y_train[-n_val:]
    X_train = X_train[:-n_val]
    y_train = y_train[:-n_val]

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_one_epoch(model, dl_train, optimizer, device):
    model.train()
    epoch_loss = 0
    for x, y in dl_train:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if isinstance(optimizer, PCGDOptimizer):
            optimizer.step(x, y)
            with torch.no_grad():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
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

    dl_train = TensorDataLoader((X_train, y_train), batch_size=32, shuffle=True)
    model = MLP().to(device)
    base_opt = torch.optim.Adam(model.parameters(), lr=lr)

    if mode == 'PC-Adam':
        optimizer = PCGDOptimizer(model, base_opt)
    else:
        optimizer = base_opt

    epochs = 5
    for epoch in range(epochs):
        train_one_epoch(model, dl_train, optimizer, device)

    acc = evaluate(model, X_val, y_val, device)
    return acc

def run_experiment(modes, X_train, y_train, X_val, y_val, X_test, y_test, device, label="clean"):
    results = {}
    best_params = {}
    histories = {}

    for mode in modes:
        print(f"\n--- Tuning {mode} ({label}) ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, mode, X_train, y_train, X_val, y_val, device), n_trials=5)

        best_params[mode] = study.best_params
        print(f"Best params for {mode}: {best_params[mode]}")

        # Final training with best params
        print(f"Final training for {mode}...")
        model = MLP().to(device)
        base_opt = torch.optim.Adam(model.parameters(), lr=best_params[mode]['lr'])
        if mode == 'PC-Adam':
            optimizer = PCGDOptimizer(model, base_opt)
        else:
            optimizer = base_opt

        dl_train = TensorDataLoader((X_train, y_train), batch_size=32, shuffle=True)

        history = {'train_loss': [], 'test_acc': []}
        epochs = 30
        for epoch in range(epochs):
            loss = train_one_epoch(model, dl_train, optimizer, device)
            test_acc = evaluate(model, X_test, y_test, device)
            history['train_loss'].append(loss)
            history['test_acc'].append(test_acc)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Test Acc: {test_acc:.4f}")

        results[mode] = evaluate(model, X_test, y_test, device)
        histories[mode] = history

    return results, histories, best_params

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Clean data experiment
    print("\n=== CLEAN DATA EXPERIMENT ===")
    X_train, y_train, X_val, y_val, X_test, y_test = get_data(noisy_fraction=0.0)
    results_clean, histories_clean, params_clean = run_experiment(['Baseline', 'PC-Adam'], X_train, y_train, X_val, y_val, X_test, y_test, device, label="clean")

    # Noisy data experiment
    print("\n=== NOISY DATA EXPERIMENT (20% noise) ===")
    X_train_n, y_train_n, X_val_n, y_val_n, X_test_n, y_test_n = get_data(noisy_fraction=0.2)
    results_noisy, histories_noisy, params_noisy = run_experiment(['Baseline', 'PC-Adam'], X_train_n, y_train_n, X_val_n, y_val_n, X_test_n, y_test_n, device, label="noisy")

    # Output results
    with open('pcgd_experiment/results.txt', 'w') as f:
        f.write("=== CLEAN DATA ===\n")
        for mode in results_clean:
            f.write(f"Mode: {mode}\n")
            f.write(f"  Best Params: {params_clean[mode]}\n")
            f.write(f"  Final Test Accuracy: {results_clean[mode]:.4f}\n")

        f.write("\n=== NOISY DATA (20%) ===\n")
        for mode in results_noisy:
            f.write(f"Mode: {mode}\n")
            f.write(f"  Best Params: {params_noisy[mode]}\n")
            f.write(f"  Final Test Accuracy: {results_noisy[mode]:.4f}\n")

    # Plotting
    plt.figure(figsize=(12, 10))

    # Clean Loss
    plt.subplot(2, 2, 1)
    for mode in results_clean:
        plt.plot(histories_clean[mode]['train_loss'], label=mode)
    plt.title('Clean Data: Training Loss')
    plt.legend()

    # Clean Acc
    plt.subplot(2, 2, 2)
    for mode in results_clean:
        plt.plot(histories_clean[mode]['test_acc'], label=mode)
    plt.title('Clean Data: Test Accuracy')
    plt.legend()

    # Noisy Loss
    plt.subplot(2, 2, 3)
    for mode in results_noisy:
        plt.plot(histories_noisy[mode]['train_loss'], label=mode)
    plt.title('Noisy Data: Training Loss')
    plt.legend()

    # Noisy Acc
    plt.subplot(2, 2, 4)
    for mode in results_noisy:
        plt.plot(histories_noisy[mode]['test_acc'], label=mode)
    plt.title('Noisy Data: Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('pcgd_experiment/comparison.png')
    print("\nExperiment complete. Results saved in pcgd_experiment/")

if __name__ == "__main__":
    main()
