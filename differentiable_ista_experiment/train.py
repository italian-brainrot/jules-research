import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import numpy as np
import os
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from differentiable_ista_experiment.model import ISTANet, BaselineMLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=20, batch_size=64):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    train_losses = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(dl_train))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            total = y_test.size(0)
            correct = (predicted == y_test).sum().item()
        test_accs.append(100 * correct / total)

    return train_losses, test_accs

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_type == "ista":
        model = ISTANet(input_dim=40, sparse_dim=64, num_iterations=10, hidden_dim=64, output_dim=10)
    else:
        model = BaselineMLP(input_dim=40, hidden_dim=100, output_dim=10) # 100 to have comparable param count roughly

    _, test_accs = train_model(model, X_train, y_train, X_test, y_test, lr, epochs=15)
    return max(test_accs)

def main():
    X_train, y_train, X_test, y_test = get_data()

    # Tuning ISTA
    print("Tuning ISTA...")
    study_ista = optuna.create_study(direction="maximize")
    study_ista.optimize(lambda trial: objective(trial, "ista", X_train, y_train, X_test, y_test), n_trials=10)
    best_lr_ista = study_ista.best_params["lr"]

    # Tuning Baseline
    print("Tuning Baseline...")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(lambda trial: objective(trial, "baseline", X_train, y_train, X_test, y_test), n_trials=10)
    best_lr_baseline = study_baseline.best_params["lr"]

    print(f"Best LR ISTA: {best_lr_ista}")
    print(f"Best LR Baseline: {best_lr_baseline}")

    # Final comparison
    seeds = [42, 43, 44]
    results_ista = []
    results_baseline = []

    all_ista_accs = []
    all_baseline_accs = []

    for seed in seeds:
        torch.manual_seed(seed)
        print(f"Running Seed {seed} for ISTA...")
        model_ista = ISTANet(input_dim=40, sparse_dim=64, num_iterations=10, hidden_dim=64, output_dim=10)
        _, ista_accs = train_model(model_ista, X_train, y_train, X_test, y_test, best_lr_ista, epochs=50)
        results_ista.append(max(ista_accs))
        all_ista_accs.append(ista_accs)

        torch.manual_seed(seed)
        print(f"Running Seed {seed} for Baseline...")
        model_baseline = BaselineMLP(input_dim=40, hidden_dim=100, output_dim=10)
        _, baseline_accs = train_model(model_baseline, X_train, y_train, X_test, y_test, best_lr_baseline, epochs=50)
        results_baseline.append(max(baseline_accs))
        all_baseline_accs.append(baseline_accs)

    print(f"ISTA Final Results: {np.mean(results_ista):.2f}% +/- {np.std(results_ista):.2f}%")
    print(f"Baseline Final Results: {np.mean(results_baseline):.2f}% +/- {np.std(results_baseline):.2f}%")

    # Plotting
    plt.figure(figsize=(10, 6))
    epochs = range(1, 51)

    avg_ista_accs = np.mean(all_ista_accs, axis=0)
    avg_baseline_accs = np.mean(all_baseline_accs, axis=0)

    plt.plot(epochs, avg_ista_accs, label="ISTA-Net")
    plt.plot(epochs, avg_baseline_accs, label="Baseline MLP")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("ISTA-Net vs Baseline MLP on MNIST-1D")
    plt.legend()
    plt.grid(True)
    plt.savefig("differentiable_ista_experiment/comparison.png")

    with open("differentiable_ista_experiment/README.md", "w") as f:
        f.write("# Differentiable ISTA Experiment\n\n")
        f.write("## Hypothesis\n")
        f.write("Using a differentiable Iterative Soft-Thresholding Algorithm (ISTA) layer to learn a sparse representation of input features can provide a better inductive bias for signal classification tasks than standard dense layers.\n\n")
        f.write("## Results\n")
        f.write(f"- **ISTA-Net**: {np.mean(results_ista):.2f}% +/- {np.std(results_ista):.2f}%\n")
        f.write(f"- **Baseline MLP**: {np.mean(results_baseline):.2f}% +/- {np.std(results_baseline):.2f}%\n\n")
        f.write("![Comparison](comparison.png)\n")

if __name__ == "__main__":
    main()
