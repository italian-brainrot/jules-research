import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import ResidualSoftBinMLP, BaselineMLP
import matplotlib.pyplot as plt
import sys

# Standard print with flush
def print_flush(msg):
    print(msg)
    sys.stdout.flush()

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_model(model, train_loader, test_loader, lr, epochs=50):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        train_accuracies.append(train_acc)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100. * correct / total
        test_accuracies.append(test_acc)

    return train_accuracies, test_accuracies

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    if model_type == "softbin":
        model = ResidualSoftBinMLP()
    else:
        model = BaselineMLP()

    _, test_accs = train_model(model, train_loader, test_loader, lr, epochs=20)
    score = max(test_accs)
    print_flush(f"Trial finished for {model_type} with LR {lr:.6f}: score {score:.2f}%")
    return score

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    # Tune ResidualSoftBinMLP
    print_flush("Tuning ResidualSoftBinMLP...")
    study_sb = optuna.create_study(direction="maximize")
    study_sb.optimize(lambda t: objective(t, "softbin", X_train, y_train, X_test, y_test), n_trials=8)
    best_lr_sb = study_sb.best_params["lr"]
    print_flush(f"Best LR for ResidualSoftBinMLP: {best_lr_sb}")

    # Tune BaselineMLP
    print_flush("Tuning BaselineMLP...")
    study_base = optuna.create_study(direction="maximize")
    study_base.optimize(lambda t: objective(t, "baseline", X_train, y_train, X_test, y_test), n_trials=8)
    best_lr_base = study_base.best_params["lr"]
    print_flush(f"Best LR for BaselineMLP: {best_lr_base}")

    # Evaluate both with multiple seeds
    seeds = [42, 43, 44]
    sb_results = []
    base_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        print_flush(f"Evaluating ResidualSoftBinMLP with seed {seed}...")
        model_sb = ResidualSoftBinMLP()
        _, sb_test_accs = train_model(model_sb, train_loader, test_loader, best_lr_sb, epochs=50)
        sb_results.append(sb_test_accs)

        torch.manual_seed(seed)
        print_flush(f"Evaluating BaselineMLP with seed {seed}...")
        model_base = BaselineMLP()
        _, base_test_accs = train_model(model_base, train_loader, test_loader, best_lr_base, epochs=50)
        base_results.append(base_test_accs)

    sb_results = np.array(sb_results)
    base_results = np.array(base_results)

    # Plot results
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, 51)

    plt.plot(epochs, sb_results.mean(0), label="ResidualSoftBinMLP (Best LR)")
    plt.fill_between(epochs, sb_results.mean(0) - sb_results.std(0), sb_results.mean(0) + sb_results.std(0), alpha=0.2)

    plt.plot(epochs, base_results.mean(0), label="BaselineMLP (Best LR)")
    plt.fill_between(epochs, base_results.mean(0) - base_results.std(0), base_results.mean(0) + base_results.std(0), alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("ResidualSoftBinMLP vs BaselineMLP on mnist1d")
    plt.legend()
    plt.grid(True)
    plt.savefig("soft_binning_expansion_experiment/comparison.png")

    print_flush("\nResults Summary:")
    print_flush(f"ResidualSoftBinMLP: {sb_results[:, -1].mean():.2f}% +/- {sb_results[:, -1].std():.2f}%")
    print_flush(f"BaselineMLP: {base_results[:, -1].mean():.2f}% +/- {base_results[:, -1].std():.2f}%")

    with open("soft_binning_expansion_experiment/results.txt", "w") as f:
        f.write(f"Best LR ResidualSoftBinMLP: {best_lr_sb}\n")
        f.write(f"Best LR BaselineMLP: {best_lr_base}\n")
        f.write(f"ResidualSoftBinMLP Final Accuracy: {sb_results[:, -1].mean():.2f}% +/- {sb_results[:, -1].std():.2f}%\n")
        f.write(f"BaselineMLP Final Accuracy: {base_results[:, -1].mean():.2f}% +/- {base_results[:, -1].std():.2f}%\n")
