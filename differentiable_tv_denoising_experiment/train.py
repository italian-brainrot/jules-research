import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import numpy as np
from model import TVDenoisingMLP, BaselineMLP
import os

def get_data(noise_std=0.0):
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])

    if noise_std > 0:
        X_train += noise_std * torch.randn_like(X_train)
        X_test += noise_std * torch.randn_like(X_test)

    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=20, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            model.eval()
            test_outputs = model(X_test)
            acc = (test_outputs.argmax(1) == y_test).float().mean().item()
            if acc > best_acc:
                best_acc = acc
    return best_acc

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256)

    if model_type == "baseline":
        model = BaselineMLP(40, hidden_dim, 10)
    elif model_type == "tv_denoising":
        n_iters = trial.suggest_int("n_iters", 5, 20)
        model = TVDenoisingMLP(40, hidden_dim, 10, n_iters=n_iters)

    return train_model(model, X_train, y_train, X_test, y_test, lr)

if __name__ == "__main__":
    noise_levels = [0.0, 0.2]
    results = {}

    for noise in noise_levels:
        print(f"--- Noise level: {noise} ---")
        X_train, y_train, X_test, y_test = get_data(noise)

        for model_name in ["baseline", "tv_denoising"]:
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda t: objective(t, model_name, X_train, y_train, X_test, y_test), n_trials=10)

            best_acc = study.best_value
            print(f"Best {model_name} Accuracy (Noise {noise}): {best_acc:.4f}")
            results[f"{model_name}_noise_{noise}"] = best_acc

            # Save results progressively
            with open("differentiable_tv_denoising_experiment/results.txt", "a") as f:
                f.write(f"{model_name}_noise_{noise}: {best_acc}\n")

    with open("differentiable_tv_denoising_experiment/results.txt", "w") as f:
        for key, val in results.items():
            f.write(f"{key}: {val}\n")
