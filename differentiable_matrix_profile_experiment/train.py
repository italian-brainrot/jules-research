import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import MPAugmentedMLP, BaselineMLP
import os

def train_model(model, dl_train, dl_test, epochs=30, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for x, y in dl_train:
            optimizer.zero_grad()
            out = model(x.float())
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dl_test:
                out = model(x.float())
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
    return best_acc

def objective_baseline(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256)

    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]), torch.tensor(data['y_test'])

    dl_train = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=256, shuffle=False)

    model = BaselineMLP(input_dim=40, hidden_dim=hidden_dim, output_dim=10)
    acc = train_model(model, dl_train, dl_test, epochs=30, lr=lr)
    return acc

def objective_mp(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
    window_size = trial.suggest_int("window_size", 3, 10)
    temperature = trial.suggest_float("temperature", 0.01, 1.0, log=True)

    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]), torch.tensor(data['y_test'])

    dl_train = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=256, shuffle=False)

    model = MPAugmentedMLP(input_dim=40, hidden_dim=hidden_dim, output_dim=10, window_size=window_size)
    model.mp_layer.temperature.data = torch.tensor(temperature)

    acc = train_model(model, dl_train, dl_test, epochs=30, lr=lr)
    return acc

if __name__ == "__main__":
    print("Tuning Baseline MLP...")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(objective_baseline, n_trials=10)

    print("Tuning MP Augmented MLP...")
    study_mp = optuna.create_study(direction="maximize")
    study_mp.optimize(objective_mp, n_trials=10)

    print(f"Best Baseline Acc: {study_baseline.best_value}")
    print(f"Best Baseline Params: {study_baseline.best_params}")
    print(f"Best MP Acc: {study_mp.best_value}")
    print(f"Best MP Params: {study_mp.best_params}")

    with open("differentiable_matrix_profile_experiment/results.txt", "w") as f:
        f.write(f"Best Baseline Acc: {study_baseline.best_value}\n")
        f.write(f"Best Baseline Params: {study_baseline.best_params}\n")
        f.write(f"Best MP Acc: {study_mp.best_value}\n")
        f.write(f"Best MP Params: {study_mp.best_params}\n")
