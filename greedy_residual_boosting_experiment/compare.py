import torch
import torch.nn as nn
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import optuna
import matplotlib.pyplot as plt
import os
import json
from model import GreedyMLP
from train import train_backprop, train_greedy_stacking, train_greedy_boosting, train_greedy_boosting_fd, evaluate

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 5000
    data = make_dataset(defaults)

    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data['x_test']).float(), torch.tensor(data['y_test']).long()

    return X_train, y_train, X_test, y_test

def objective(trial, X_train, y_train, X_test, y_test, method, device):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lambd = 0
    if method == 'boosting_fd':
        lambd = trial.suggest_float("lambd", 1e-6, 1.0, log=True)

    input_dim = X_train.shape[1]
    hidden_dim = 256
    num_classes = 10
    num_layers = 3

    model = GreedyMLP(input_dim, hidden_dim, num_classes, num_layers).to(device)
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

    if method == 'backprop':
        train_backprop(model, dl_train, lr, epochs=100, device=device)
    elif method == 'stacking':
        train_greedy_stacking(model, dl_train, lr, epochs_per_layer=33, device=device)
    elif method == 'boosting':
        train_greedy_boosting(model, dl_train, lr, epochs_per_layer=33, device=device)
    elif method == 'boosting_fd':
        train_greedy_boosting_fd(model, dl_train, lr, lambd, epochs_per_layer=33, device=device)

    acc = evaluate(model, X_test, y_test, device, method=method)
    return acc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test = get_data()

    results = {}
    methods = ['backprop', 'stacking', 'boosting', 'boosting_fd']

    for method in methods:
        print(f"Optimizing {method}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, X_train, y_train, X_test, y_test, method, device), n_trials=10)
        results[method] = study.best_value
        print(f"Best accuracy for {method}: {study.best_value:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.ylabel("Test Accuracy")
    plt.title("Greedy Layer-wise Training Comparison on MNIST1D")
    plt.ylim(0, 1)
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.savefig("greedy_residual_boosting_experiment/results.png")

    with open("greedy_residual_boosting_experiment/results.json", "w") as f:
        json.dump(results, f, indent=4)
