import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import numpy as np
import optuna
import matplotlib.pyplot as plt
import os
import time
from model import OptiReLU, train_optirelu

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 2000
    data = make_dataset(defaults)

    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data['x_test']).float(), torch.tensor(data['y_test']).long()

    return X_train, y_train, X_test, y_test

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(X, y, X_test, y_test, hidden_dim, lr, epochs=200):
    model = SimpleMLP(X.shape[1], hidden_dim, 10).to(X.device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dl = TensorDataLoader((X, y), batch_size=64, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for bx, by in dl:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        acc = (test_logits.argmax(1) == y_test).float().mean().item()
    return acc

def train_linear(X, y, X_test, y_test):
    model = nn.Linear(X.shape[1], 10).to(X.device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dl = TensorDataLoader((X, y), batch_size=64, shuffle=True)
    for _ in range(100):
        for bx, by in dl:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(1) == y_test).float().mean().item()
    return acc

def train_rf(X, y, X_test, y_test, num_features, lambda_reg):
    # Random Features baseline
    input_dim = X.shape[1]
    W = torch.randn(num_features, input_dim, device=X.device)
    b = torch.randn(num_features, device=X.device)
    # Normalize
    norms = torch.norm(torch.cat([W, b.unsqueeze(1)], dim=1), dim=1, keepdim=True)
    W /= norms
    b /= norms.squeeze()

    with torch.no_grad():
        phi_X = F.relu(F.linear(X, W, b))
        phi_X_test = F.relu(F.linear(X_test, W, b))

    model = nn.Linear(num_features, 10).to(X.device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(200):
        optimizer.zero_grad()
        logits = model(phi_X)
        loss = F.cross_entropy(logits, y) + lambda_reg * torch.norm(model.weight, p=2, dim=0).sum()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        acc = (model(phi_X_test).argmax(1) == y_test).float().mean().item()
    return acc

def objective_mlp(trial, X, y, X_test, y_test, hidden_dim):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    return train_mlp(X, y, X_test, y_test, hidden_dim, lr)

def objective_optirelu(trial, X, y, X_test, y_test, max_neurons):
    lambda_reg = trial.suggest_float("lambda_reg", 1e-5, 1e-1, log=True)
    model = OptiReLU(X.shape[1], 10, lambda_reg=lambda_reg).to(X.device)
    model = train_optirelu(model, X, y, max_neurons=max_neurons, lambda_reg=lambda_reg)

    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(1) == y_test).float().mean().item()
    return acc

def objective_rf(trial, X, y, X_test, y_test, num_features):
    lambda_reg = trial.suggest_float("lambda_reg", 1e-5, 1e-1, log=True)
    return train_rf(X, y, X_test, y_test, num_features, lambda_reg)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test = get_data()
    X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)

    results = {}

    # Linear Baseline
    print("Training Linear Baseline...")
    results["Linear"] = train_linear(X_train, y_train, X_test, y_test)
    print(f"Linear Acc: {results['Linear']:.4f}")

    # Target number of neurons
    target_neurons = 50

    # OptiReLU
    print(f"Training OptiReLU (Target Max {target_neurons} neurons)...")
    study_optirelu = optuna.create_study(direction="maximize")
    study_optirelu.optimize(lambda t: objective_optirelu(t, X_train, y_train, X_test, y_test, target_neurons), n_trials=10)
    best_lambda = study_optirelu.best_params["lambda_reg"]

    # Run once more with best lambda to get final model
    opti_model = OptiReLU(X_train.shape[1], 10, lambda_reg=best_lambda).to(device)
    opti_model = train_optirelu(opti_model, X_train, y_train, max_neurons=target_neurons, lambda_reg=best_lambda)
    with torch.no_grad():
        results["OptiReLU"] = (opti_model(X_test).argmax(1) == y_test).float().mean().item()
    n_opti = opti_model.hidden_weights.shape[0]
    print(f"OptiReLU Acc: {results['OptiReLU']:.4f} (Neurons: {n_opti})")

    # MLP Baseline (same width as OptiReLU final width)
    print(f"Training MLP Baseline (Width {n_opti})...")
    study_mlp = optuna.create_study(direction="maximize")
    study_mlp.optimize(lambda t: objective_mlp(t, X_train, y_train, X_test, y_test, n_opti), n_trials=10)
    results["MLP"] = study_mlp.best_value
    print(f"MLP Acc: {results['MLP']:.4f}")

    # Random Features Baseline
    print(f"Training Random Features Baseline (Width {n_opti})...")
    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(lambda t: objective_rf(t, X_train, y_train, X_test, y_test, n_opti), n_trials=10)
    results["RandomFeatures"] = study_rf.best_value
    print(f"RF Acc: {results['RandomFeatures']:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), results.values())
    plt.ylabel("Test Accuracy")
    plt.title(f"Comparison on MNIST1D (OptiReLU neurons: {n_opti})")
    plt.ylim(0, 1)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')
    plt.savefig("opti_relu_experiment/comparison.png")

    # Save results to file
    with open("opti_relu_experiment/results.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write(f"OptiReLU Final Neurons: {n_opti}\n")
        f.write(f"Best OptiReLU Lambda: {best_lambda}\n")
