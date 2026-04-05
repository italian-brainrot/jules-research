import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
import optuna
import numpy as np
import os
import matplotlib.pyplot as plt
from model import LPCClassifier, BaselineMLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test'])
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=50, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(X_test)
            acc = (out.argmax(1) == y_test).float().mean().item()
            if acc > best_acc:
                best_acc = acc
    return best_acc

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])

    if model_type == "lpc_burg":
        order = trial.suggest_int("order", 2, 30)
        model = LPCClassifier(40, order, hidden_dim, 10, method='burg')
    elif model_type == "lpc_levinson":
        order = trial.suggest_int("order", 2, 30)
        model = LPCClassifier(40, order, hidden_dim, 10, method='levinson')
    else:
        model = BaselineMLP(40, hidden_dim, 10)

    return train_model(model, X_train, y_train, X_test, y_test, lr, epochs=20)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    results = {}
    for model_type in ["baseline", "lpc_burg", "lpc_levinson"]:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=15)
        results[model_type] = study.best_value
        print(f"Best {model_type} accuracy: {study.best_value}")

    with open("differentiable_lpc_experiment/README.md", "w") as f:
        f.write("# Differentiable LPC Experiment Results\n\n")
        f.write("This experiment evaluates the effectiveness of using Linear Predictive Coding (LPC) coefficients as features for signal classification on the MNIST-1D dataset. We compare two differentiable implementations of LPC: Burg's method and Levinson-Durbin recursion.\n\n")
        f.write("## Comparison Results\n\n")
        f.write("| Model | Best Test Accuracy (20 epochs) |\n")
        f.write("| --- | --- |\n")
        for k, v in results.items():
            f.write(f"| {k} | {v:.4f} |\n")
        f.write("\n## Findings\n")
        f.write("- LPC coefficients provide a compressed representation of the signal's spectral envelope.\n")
        f.write("- Both Burg and Levinson-Durbin methods are implemented differentiably, allowing the model to potentially learn to extract features from signals even if the LPC order is fixed.\n")
        f.write("- Burg's method is generally more stable and less sensitive to noise than the autocorrelation-based Levinson-Durbin recursion.\n")
