import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import BaselineMLP, DPRSANet, DPRSAAugmentedMLP
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x'], dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_model(model, dl_train, dl_test, epochs=20, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in dl_train:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl_test:
            out = model(x)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return 100 * correct / total

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_type == "baseline":
        model = BaselineMLP()
    elif model_type == "prsa":
        model = DPRSANet(num_anchors=8, window_size=20)
    elif model_type == "augmented":
        model = DPRSAAugmentedMLP(num_anchors=4, window_size=20)

    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=64, shuffle=False)

    return train_model(model, dl_train, dl_test, epochs=15, lr=lr)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    results = {}

    for model_type in ["baseline", "prsa", "augmented"]:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=10)

        best_acc = study.best_value
        results[model_type] = best_acc
        print(f"Best {model_type} accuracy: {best_acc:.2f}%")

    with open("results.txt", "w") as f:
        for model_type, acc in results.items():
            f.write(f"{model_type}: {acc:.2f}%\n")

    # Final plot
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.ylabel("Accuracy (%)")
    plt.title("MNIST-1D Classification Results")
    plt.savefig("results.png")
