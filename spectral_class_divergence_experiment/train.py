import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import optuna
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.func import vmap, grad

def compute_input_gradients(model, inputs, targets):
    def loss_fn(x, y):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        output = model(x)
        loss = F.cross_entropy(output, y)
        return loss
    return vmap(grad(loss_fn))(inputs, targets)

def compute_icsd_penalty(input_gradients, targets, num_classes=10):
    B, L = input_gradients.shape
    freqs = torch.fft.rfft(input_gradients, dim=1)
    power = torch.abs(freqs)**2
    power_norm = power / (power.sum(dim=1, keepdim=True) + 1e-8)

    class_spectra = []
    valid_classes = []
    for c in range(num_classes):
        mask = (targets == c)
        if mask.any():
            class_mean = power_norm[mask].mean(dim=0)
            class_spectra.append(class_mean)
            valid_classes.append(c)

    if len(class_spectra) < 2:
        return torch.tensor(0.0, device=input_gradients.device)

    class_spectra = torch.stack(class_spectra)
    dot_prod = torch.mm(class_spectra, class_spectra.t())
    norms = torch.norm(class_spectra, dim=1)
    norm_prod = torch.outer(norms, norms)
    cos_sim = dot_prod / (norm_prod + 1e-8)

    mask = torch.eye(len(valid_classes), device=input_gradients.device).bool()
    cos_sim_others = cos_sim[~mask]

    return cos_sim_others.mean()

class MLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_epoch(model, dl, optimizer, lambda_icsd=0.0):
    model.train()
    total_loss = 0
    total_ce = 0
    total_icsd = 0
    correct = 0
    total = 0

    for inputs, targets in dl:
        optimizer.zero_grad()

        if lambda_icsd > 0:
            # We need to calculate input gradients, so we need inputs to require grad
            inputs.requires_grad_(True)
            input_grads = compute_input_gradients(model, inputs, targets)
            icsd_penalty = compute_icsd_penalty(input_grads, targets)
        else:
            icsd_penalty = torch.tensor(0.0, device=inputs.device)

        outputs = model(inputs)
        ce_loss = F.cross_entropy(outputs, targets)

        loss = ce_loss + lambda_icsd * icsd_penalty
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_ce += ce_loss.item() * inputs.size(0)
        total_icsd += icsd_penalty.item() * inputs.size(0)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, total_ce / total, total_icsd / total, correct / total

def validate(model, dl):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dl:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()

    # Val split
    idx = torch.randperm(len(X_train))
    train_size = int(0.8 * len(X_train))
    train_idx, val_idx = idx[:train_size], idx[train_size:]

    dl_train = TensorDataLoader((X_train[train_idx], y_train[train_idx]), batch_size=128, shuffle=True)
    dl_val = TensorDataLoader((X_train[val_idx], y_train[val_idx]), batch_size=128, shuffle=False)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    return dl_train, dl_val, dl_test

def objective(trial, mode):
    dl_train, dl_val, dl_test = get_data()
    model = MLP()
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

    lambda_icsd = 0.0
    if mode == "icsd":
        lambda_icsd = trial.suggest_float("lambda_icsd", 1e-4, 1e1, log=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0
    for epoch in range(20):
        train_epoch(model, dl_train, optimizer, lambda_icsd)
        val_acc = validate(model, dl_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tune", "eval"], default="tune")
    args = parser.parse_args()

    if args.mode == "tune":
        params = {}
        for mode in ["baseline", "icsd"]:
            study = optuna.create_study(direction="maximize")
            n_trials = 15 if mode == "baseline" else 10
            study.optimize(lambda trial: objective(trial, mode), n_trials=n_trials)
            params[mode] = study.best_params
            print(f"Best params for {mode}: {study.best_params}")

        with open("spectral_class_divergence_experiment/best_params.json", "w") as f:
            json.dump(params, f)

    elif args.mode == "eval":
        if not os.path.exists("spectral_class_divergence_experiment/best_params.json"):
            print("Please run tuning first.")
            return

        with open("spectral_class_divergence_experiment/best_params.json", "r") as f:
            params = json.load(f)

        results = {}
        dl_train, dl_val, dl_test = get_data()

        for mode in ["baseline", "icsd"]:
            print(f"Evaluating {mode}...")
            test_accs = []
            histories = []

            for seed in range(5):
                torch.manual_seed(seed)
                model = MLP()
                mode_params = params[mode]
                optimizer = optim.AdamW(model.parameters(), lr=mode_params["lr"], weight_decay=mode_params["weight_decay"])
                lambda_icsd = mode_params.get("lambda_icsd", 0.0)

                history = {"train_loss": [], "train_acc": [], "test_acc": []}
                best_test_acc = 0
                for epoch in range(50):
                    train_loss, train_ce, train_icsd, train_acc = train_epoch(model, dl_train, optimizer, lambda_icsd)
                    test_acc = validate(model, dl_test)
                    history["train_loss"].append(train_loss)
                    history["train_acc"].append(train_acc)
                    history["test_acc"].append(test_acc)
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                    if (epoch + 1) % 10 == 0:
                        print(f"Seed {seed}, Epoch {epoch+1}, Test Acc: {test_acc:.4f}")

                test_accs.append(best_test_acc)
                histories.append(history)

            results[mode] = {
                "mean_test_acc": np.mean(test_accs),
                "std_test_acc": np.std(test_accs),
                "histories": histories
            }
            print(f"Mode {mode}: {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}")

        with open("spectral_class_divergence_experiment/results.json", "w") as f:
            # We need a custom encoder for the lists/arrays if they were numpy, but they are standard lists.
            json.dump(results, f)

        # Plotting
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for mode in ["baseline", "icsd"]:
            all_train_acc = np.array([h["train_acc"] for h in results[mode]["histories"]])
            plt.plot(all_train_acc.mean(axis=0), label=f"{mode} Train")
        plt.title("Training Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        for mode in ["baseline", "icsd"]:
            all_test_acc = np.array([h["test_acc"] for h in results[mode]["histories"]])
            plt.plot(all_test_acc.mean(axis=0), label=f"{mode} Test")
        plt.title("Test Accuracy")
        plt.legend()
        plt.savefig("spectral_class_divergence_experiment/results.png")

if __name__ == "__main__":
    main()
