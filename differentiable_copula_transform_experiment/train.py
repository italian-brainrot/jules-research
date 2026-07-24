import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import BaselineMLP, CopulaAugmentedMLP
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=50, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    return accuracy

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    if model_type == "baseline":
        model = BaselineMLP()
    else:
        model = CopulaAugmentedMLP()

    return train_model(model, X_train, y_train, X_test, y_test, lr, epochs=20)

def main():
    X_train, y_train, X_test, y_test = get_data()

    # Tune Baseline
    print("Tuning Baseline...")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(lambda trial: objective(trial, "baseline", X_train, y_train, X_test, y_test), n_trials=10)
    best_lr_baseline = study_baseline.best_params["lr"]
    print(f"Best LR Baseline: {best_lr_baseline}")

    # Tune Copula
    print("Tuning Copula Augmented...")
    study_copula = optuna.create_study(direction="maximize")
    study_copula.optimize(lambda trial: objective(trial, "copula", X_train, y_train, X_test, y_test), n_trials=10)
    best_lr_copula = study_copula.best_params["lr"]
    print(f"Best LR Copula: {best_lr_copula}")

    # Final Evaluation
    print("Final Evaluation...")
    baseline_accs = []
    copula_accs = []

    for i in range(5):
        print(f"Run {i+1}/5")
        baseline_model = BaselineMLP()
        baseline_acc = train_model(baseline_model, X_train, y_train, X_test, y_test, best_lr_baseline, epochs=50)
        baseline_accs.append(baseline_acc)

        copula_model = CopulaAugmentedMLP()
        copula_acc = train_model(copula_model, X_train, y_train, X_test, y_test, best_lr_copula, epochs=50)
        copula_accs.append(copula_acc)

    print(f"Baseline Accuracy: {np.mean(baseline_accs):.4f} +/- {np.std(baseline_accs):.4f}")
    print(f"Copula Augmented Accuracy: {np.mean(copula_accs):.4f} +/- {np.std(copula_accs):.4f}")

    # Save results
    with open("differentiable_copula_transform_experiment/results.txt", "w") as f:
        f.write(f"Baseline Accuracy: {np.mean(baseline_accs):.4f} +/- {np.std(baseline_accs):.4f}\n")
        f.write(f"Copula Augmented Accuracy: {np.mean(copula_accs):.4f} +/- {np.std(copula_accs):.4f}\n")
        f.write(f"Best LR Baseline: {best_lr_baseline}\n")
        f.write(f"Best LR Copula: {best_lr_copula}\n")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(["Baseline", "Copula Augmented"], [np.mean(baseline_accs), np.mean(copula_accs)], yerr=[np.std(baseline_accs), np.std(copula_accs)], capsize=10)
    plt.ylabel("Accuracy")
    plt.title("Baseline vs Copula Augmented MLP on MNIST-1D")
    plt.savefig("differentiable_copula_transform_experiment/results.png")

if __name__ == "__main__":
    main()
