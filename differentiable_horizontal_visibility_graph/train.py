import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import BaselineMLP, DHVGAugmentedMLP, DHVGGNN
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 2000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=50, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_test.to(device))
            acc = (outputs.argmax(dim=1) == y_test.to(device)).float().mean().item()
            if acc > best_acc:
                best_acc = acc
    return best_acc

def objective(trial, model_name, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    if model_name == "baseline":
        model = BaselineMLP()
    elif model_name == "dhvg_aug":
        model = DHVGAugmentedMLP()
    elif model_name == "dhvg_gnn":
        model = DHVGGNN()

    return train_model(model, X_train, y_train, X_test, y_test, lr, epochs=20)

def main():
    X_train, y_train, X_test, y_test = get_data()

    results = {}
    best_lrs = {}

    for model_name in ["baseline", "dhvg_aug", "dhvg_gnn"]:
        print(f"Tuning {model_name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_name, X_train, y_train, X_test, y_test), n_trials=3)
        best_lrs[model_name] = study.best_params["lr"]
        print(f"Best LR for {model_name}: {best_lrs[model_name]}")

        # Evaluate with 1 seed
        accs = []
        for seed in [0]:
            torch.manual_seed(seed)
            if model_name == "baseline":
                model = BaselineMLP()
            elif model_name == "dhvg_aug":
                model = DHVGAugmentedMLP()
            elif model_name == "dhvg_gnn":
                model = DHVGGNN()

            acc = train_model(model, X_train, y_train, X_test, y_test, best_lrs[model_name], epochs=20)
            accs.append(acc)
        results[model_name] = accs
        print(f"{model_name} accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    # Save results
    with open("differentiable_horizontal_visibility_graph/results.txt", "w") as f:
        for model_name, accs in results.items():
            f.write(f"{model_name}: {np.mean(accs):.4f} +/- {np.std(accs):.4f} (Best LR: {best_lrs[model_name]})\n")

    # Plotting
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    means = [np.mean(results[name]) for name in model_names]
    stds = [np.std(results[name]) for name in model_names]
    plt.bar(model_names, means, yerr=stds, capsize=5)
    plt.ylabel("Accuracy")
    plt.title("MNIST-1D Classification with DHVG")
    plt.savefig("differentiable_horizontal_visibility_graph/results.png")

if __name__ == "__main__":
    main()
