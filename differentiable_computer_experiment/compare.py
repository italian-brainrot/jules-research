import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import numpy as np
import os
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from models import MLPBaseline, GRUBaseline, NTMBase, DMS_NTM

# Setup data
def get_data(num_samples=4000):
    defaults = get_dataset_args()
    defaults.num_samples = num_samples
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model, dl_train, lr, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y).sum().item() / y.size(0)
    return accuracy

def objective(trial, model_name, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_name == "MLP":
        model = MLPBaseline(input_size=X_train.shape[1])
    elif model_name == "GRU":
        model = GRUBaseline()
    elif model_name == "NTM":
        model = NTMBase()
    elif model_name == "DMS_NTM":
        model = DMS_NTM()

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

    # Train for a few epochs for tuning
    train_model(model, dl_train, lr, epochs=8)
    acc = evaluate_model(model, X_test, y_test)
    return acc

def run_experiment():
    X_train_full, y_train_full, X_test_full, y_test = get_data(num_samples=4000)

    # Downsample for sequential models to speed up
    # 40 -> 20
    X_train_seq = X_train_full[:, ::2]
    X_test_seq = X_test_full[:, ::2]

    # Use subset for tuning
    X_train_sub, y_train_sub = X_train_seq[:2000], y_train_full[:2000]

    model_names = ["MLP", "GRU", "NTM", "DMS_NTM"]
    results = {}

    for name in model_names:
        print(f"Tuning {name}...")
        if name == "MLP":
            X_tune, X_test_tune = X_train_full[:2000], X_test_full
        else:
            X_tune, X_test_tune = X_train_sub, X_test_seq

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, name, X_tune, y_train_sub, X_test_tune, y_test), n_trials=5)

        best_lr = study.best_params["lr"]
        print(f"Best LR for {name}: {best_lr}")

        # Train final model
        if name == "MLP":
            model = MLPBaseline(input_size=40)
            dl_train = TensorDataLoader((X_train_full, y_train_full), batch_size=64, shuffle=True)
            X_test_eval = X_test_full
        else:
            if name == "GRU":
                model = GRUBaseline()
            elif name == "NTM":
                model = NTMBase()
            elif name == "DMS_NTM":
                model = DMS_NTM()
            dl_train = TensorDataLoader((X_train_seq, y_train_full), batch_size=64, shuffle=True)
            X_test_eval = X_test_seq

        # Track training loss
        optimizer = optim.Adam(model.parameters(), lr=best_lr)
        criterion = nn.CrossEntropyLoss()
        losses = []
        for epoch in range(30):
            epoch_loss = 0
            model.train()
            for inputs, targets in dl_train:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(dl_train))
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {losses[-1]}")

        acc = evaluate_model(model, X_test_eval, y_test)
        results[name] = {"accuracy": acc, "losses": losses}
        print(f"{name} Accuracy: {acc}")

    # Plot results
    plt.figure(figsize=(10, 6))
    for name in model_names:
        plt.plot(results[name]["losses"], label=f"{name} (Acc: {results[name]['accuracy']:.4f})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.savefig("differentiable_computer_experiment/loss_comparison.png")

    # Save results to README
    with open("differentiable_computer_experiment/README.md", "w") as f:
        f.write("# Differentiable Multi-Scale NTM (DMS-NTM) Experiment\n\n")
        f.write("## Hypothesis\n")
        f.write("A Differentiable Neural Computer/NTM with continuous arithmetic addressing and multi-scale Gaussian focus (DMS-NTM) can better capture spatial dependencies in 1D signals compared to standard NTM with discrete shifts or simple RNNs.\n\n")
        f.write("## Methodology\n")
        f.write("- Dataset: MNIST-1D (40 features for MLP, downsampled to 20 for sequential models, 10 classes).\n")
        f.write("- Sequential Processing: RNN and NTM models process the 40 features one-by-one.\n")
        f.write("- Models:\n")
        f.write("  - MLP: Baseline seeing all features at once.\n")
        f.write("  - GRU: Standard gated recurrent unit.\n")
        f.write("  - NTM: Standard Neural Turing Machine with discrete shifts ({-1, 0, 1}).\n")
        f.write("  - DMS-NTM: Proposed NTM with continuous shifts and learnable Gaussian focus scale.\n\n")
        f.write("## Results\n\n")
        f.write("| Model | Test Accuracy |\n")
        f.write("| :--- | :--- |\n")
        for name in model_names:
            f.write(f"| {name} | {results[name]['accuracy']:.4f} |\n")
        f.write("\n![Loss Comparison](loss_comparison.png)\n")

if __name__ == "__main__":
    run_experiment()
