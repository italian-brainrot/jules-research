import torch
import torch.nn as nn
import torch.optim as optim
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
import optuna
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from music import MUSICMLP, MUSICAugmentedMLP, BaselineMLP

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
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])

    if model_type == "music":
        window_size = trial.suggest_int("window_size", 5, 20)
        num_freqs = trial.suggest_categorical("num_freqs", [32, 64])
        model = MUSICMLP(40, window_size, hidden_dim, 10, num_freqs=num_freqs)
    elif model_type == "music_augmented":
        window_size = trial.suggest_int("window_size", 5, 20)
        num_freqs = trial.suggest_categorical("num_freqs", [32, 64])
        model = MUSICAugmentedMLP(40, window_size, hidden_dim, 10, num_freqs=num_freqs)
    else:
        model = BaselineMLP(40, hidden_dim, 10)

    return train_model(model, X_train, y_train, X_test, y_test, lr, epochs=20)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    results = {}
    best_params = {}

    model_types = ["baseline", "music", "music_augmented"]

    for model_type in model_types:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=20)
        results[model_type] = study.best_value
        best_params[model_type] = study.best_params
        print(f"Best {model_type} accuracy: {study.best_value}")

    # Save results
    os.makedirs("differentiable_music_experiment", exist_ok=True)
    with open("differentiable_music_experiment/results.json", "w") as f:
        json.dump({"results": results, "best_params": best_params}, f, indent=4)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.ylabel("Accuracy")
    plt.title("Model Comparison on MNIST-1D")
    plt.savefig("differentiable_music_experiment/comparison.png")

    # Generate example MUSIC pseudospectrum
    model_type = "music"
    params = best_params[model_type]
    music_layer = MUSICMLP(40, params['window_size'], params['hidden_dim'], 10, num_freqs=params['num_freqs']).music

    plt.figure(figsize=(12, 8))
    for i in range(5):
        plt.subplot(5, 2, 2*i + 1)
        plt.plot(X_test[i].numpy())
        plt.title(f"Signal (Class {y_test[i].item()})")

        plt.subplot(5, 2, 2*i + 2)
        with torch.no_grad():
            ps = music_layer(X_test[i].unsqueeze(0)).squeeze(0).numpy()
        plt.plot(ps)
        plt.title("MUSIC Pseudospectrum")

    plt.tight_layout()
    plt.savefig("differentiable_music_experiment/pseudospectra.png")
