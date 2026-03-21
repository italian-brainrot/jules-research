import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import LRQIN, BaselineMLP
import os
import json

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model, dl_train, dl_test, lr, epochs=50, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        train_accs.append(correct / total)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dl_test:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = out.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        test_accs.append(correct / total)

    return train_accs, test_accs

def objective(trial, model_name, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_name == "LRQIN":
        model = LRQIN()
    else:
        model = BaselineMLP()

    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=64, shuffle=False)

    _, test_accs = train_model(model, dl_train, dl_test, lr, epochs=20)
    return max(test_accs)

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {}

    for model_name in ["LRQIN", "BaselineMLP"]:
        print(f"Tuning {model_name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_name, X_train, y_train, X_test, y_test), n_trials=10)

        best_lr = study.best_params["lr"]
        print(f"Best LR for {model_name}: {best_lr}")

        all_test_accs = []
        all_train_accs = []

        for seed in range(5):
            print(f"Evaluating {model_name} (Seed {seed})...")
            torch.manual_seed(seed)
            if model_name == "LRQIN":
                model = LRQIN()
            else:
                model = BaselineMLP()

            dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
            dl_test = TensorDataLoader((X_test, y_test), batch_size=64, shuffle=False)

            train_accs, test_accs = train_model(model, dl_train, dl_test, best_lr, epochs=50, device=device)
            all_train_accs.append(train_accs)
            all_test_accs.append(test_accs)

        results[model_name] = {
            "best_lr": best_lr,
            "train_accs": all_train_accs,
            "test_accs": all_test_accs,
            "final_mean": np.mean([accs[-1] for accs in all_test_accs]),
            "final_std": np.std([accs[-1] for accs in all_test_accs])
        }

    # Plotting
    plt.figure(figsize=(10, 6))
    for model_name, res in results.items():
        mean_test = np.mean(res["test_accs"], axis=0)
        std_test = np.std(res["test_accs"], axis=0)
        plt.plot(mean_test, label=f"{model_name} (Mean Test Acc: {res['final_mean']:.4f})")
        plt.fill_between(range(len(mean_test)), mean_test - std_test, mean_test + std_test, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("LRQIN vs BaselineMLP on MNIST-1D")
    plt.legend()
    plt.grid(True)
    plt.savefig("quadratic_interaction_network/comparison_plot.png")

    # Save results to JSON for README
    with open("quadratic_interaction_network/results.json", "w") as f:
        json.dump({k: {
            "best_lr": v["best_lr"],
            "final_mean": v["final_mean"],
            "final_std": v["final_std"]
        } for k, v in results.items()}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
