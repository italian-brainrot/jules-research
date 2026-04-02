import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import ChebyshevConvMLP, BaselineConvMLP
import matplotlib.pyplot as plt
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.int64)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.int64)

    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=50, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    train_losses = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            total = y_test.size(0)
            correct = (predicted == y_test).sum().item()
        test_accs.append(correct / total)

    return max(test_accs), train_losses, test_accs

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_type == "cheby":
        model = ChebyshevConvMLP()
    else:
        model = BaselineConvMLP()

    acc, _, _ = train_model(model, X_train, y_train, X_test, y_test, lr, epochs=20)
    return acc

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    print("Tuning Baseline model...")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(lambda t: objective(t, "baseline", X_train, y_train, X_test, y_test), n_trials=5)
    best_lr_baseline = study_baseline.best_params["lr"]

    print("Tuning Chebyshev model...")
    study_cheby = optuna.create_study(direction="maximize")
    study_cheby.optimize(lambda t: objective(t, "cheby", X_train, y_train, X_test, y_test), n_trials=5)
    best_lr_cheby = study_cheby.best_params["lr"]

    print(f"Best LR Baseline: {best_lr_baseline}")
    print(f"Best LR Chebyshev: {best_lr_cheby}")

    # Final evaluation across 3 seeds
    baseline_accs = []
    cheby_accs = []

    for seed in range(3):
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"Running seed {seed}...")

        # Baseline
        model_b = BaselineConvMLP()
        acc_b, loss_b, accs_b = train_model(model_b, X_train, y_train, X_test, y_test, best_lr_baseline, epochs=40)
        baseline_accs.append(acc_b)

        # Chebyshev
        model_c = ChebyshevConvMLP()
        acc_c, loss_c, accs_c = train_model(model_c, X_train, y_train, X_test, y_test, best_lr_cheby, epochs=40)
        cheby_accs.append(acc_c)

    mean_baseline = np.mean(baseline_accs)
    std_baseline = np.std(baseline_accs)
    mean_cheby = np.mean(cheby_accs)
    std_cheby = np.std(cheby_accs)

    print(f"Baseline Accuracy: {mean_baseline:.4f} +/- {std_baseline:.4f}")
    print(f"Chebyshev Accuracy: {mean_cheby:.4f} +/- {std_cheby:.4f}")

    with open("differentiable_chebyshev_filter_experiment/results.txt", "w") as f:
        f.write(f"Baseline Accuracy: {mean_baseline:.4f} +/- {std_baseline:.4f}\n")
        f.write(f"Chebyshev Accuracy: {mean_cheby:.4f} +/- {std_cheby:.4f}\n")
        f.write(f"Baseline Accs: {baseline_accs}\n")
        f.write(f"Chebyshev Accs: {cheby_accs}\n")

    # Final training curves plot for the last seed
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_b, label="Baseline Loss")
    plt.plot(loss_c, label="Chebyshev Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accs_b, label="Baseline Test Acc")
    plt.plot(accs_c, label="Chebyshev Test Acc")
    plt.title("Test Accuracy")
    plt.legend()

    plt.savefig("differentiable_chebyshev_filter_experiment/results.png")
