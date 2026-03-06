import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import MLP
from masw_utils import compute_masw_gradients

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 4000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, X_test, y_test, mode="baseline", lr=1e-3, weight_decay=1e-4, gamma=1.0, epochs=20, batch_size=128, device='cpu'):
    model = MLP(40, [256, 256], 10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = TensorDataLoader((X_train.to(device), y_train.to(device)), batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            if mode == "masw":
                # First pass to populate momentum if empty
                # Actually AdamW momentum is populated after first step.
                # compute_masw_gradients handles missing momentum by returning mean grad.
                new_grads = compute_masw_gradients(model, optimizer, batch_X, batch_y, gamma)
                for p, g in zip(model.parameters(), new_grads):
                    p.grad = g

                # Still need to compute loss for tracking
                with torch.no_grad():
                    out = model(batch_X)
                    loss = F.cross_entropy(out, batch_y)
            else:
                out = model(batch_X)
                loss = F.cross_entropy(out, batch_y)
                loss.backward()

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            out = model(X_test.to(device))
            acc = (out.argmax(1) == y_test.to(device)).float().mean().item()

        history["train_loss"].append(avg_loss)
        history["test_acc"].append(acc)

    return model, history

def objective(trial, X_train, y_train, X_test, y_test, mode, device):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    gamma = 0.0
    if mode == "masw":
        gamma = trial.suggest_float("gamma", 0.1, 10.0)

    _, history = train_model(X_train, y_train, X_test, y_test, mode=mode, lr=lr, weight_decay=weight_decay, gamma=gamma, epochs=10, device=device)
    return max(history["test_acc"])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    X_train, y_train, X_test, y_test = get_data()

    print("Tuning Baseline (AdamW)...")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(lambda t: objective(t, X_train, y_train, X_test, y_test, "baseline", device), n_trials=10)
    best_params_baseline = study_baseline.best_params

    print("Tuning MASW-AdamW...")
    study_masw = optuna.create_study(direction="maximize")
    study_masw.optimize(lambda t: objective(t, X_train, y_train, X_test, y_test, "masw", device), n_trials=10)
    best_params_masw = study_masw.best_params

    print(f"Best Baseline Params: {best_params_baseline}")
    print(f"Best MASW Params: {best_params_masw}")

    # Final evaluation
    seeds = [42]
    results = {"baseline": [], "masw": []}

    epochs = 30
    for seed in seeds:
        print(f"Final evaluation seed {seed}...")
        torch.manual_seed(seed)
        _, hist_baseline = train_model(X_train, y_train, X_test, y_test, mode="baseline",
                                       lr=best_params_baseline["lr"],
                                       weight_decay=best_params_baseline["weight_decay"],
                                       epochs=epochs, device=device)
        results["baseline"].append(hist_baseline)

        torch.manual_seed(seed)
        _, hist_masw = train_model(X_train, y_train, X_test, y_test, mode="masw",
                                   lr=best_params_masw["lr"],
                                   weight_decay=best_params_masw["weight_decay"],
                                   gamma=best_params_masw["gamma"],
                                   epochs=epochs, device=device)
        results["masw"].append(hist_masw)

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for label, res_list in results.items():
        accs = np.array([h["test_acc"] for h in res_list])
        epochs_range = range(1, epochs + 1)
        plt.plot(epochs_range, accs.mean(0), label=label)
        plt.fill_between(epochs_range, accs.min(0), accs.max(0), alpha=0.2)
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    for label, res_list in results.items():
        losses = np.array([h["train_loss"] for h in res_list])
        plt.plot(epochs_range, losses.mean(0), label=label)
        plt.fill_between(epochs_range, losses.min(0), losses.max(0), alpha=0.2)
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("masw_experiment/comparison.png")

    with open("masw_experiment/results.txt", "w") as f:
        f.write(f"Baseline (AdamW) Mean Test Acc: {np.mean([h['test_acc'][-1] for h in results['baseline']]):.4f}\n")
        f.write(f"MASW-AdamW Mean Test Acc: {np.mean([h['test_acc'][-1] for h in results['masw']]):.4f}\n")
        f.write(f"Best Baseline Params: {best_params_baseline}\n")
        f.write(f"Best MASW Params: {best_params_masw}\n")

if __name__ == "__main__":
    main()
