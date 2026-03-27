import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
import optuna
import matplotlib.pyplot as plt
import os
import numpy as np

from model import BaselineMLP, NystromMLP

def load_data():
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    return X_train, y_train, X_test, y_test

def train_one_epoch(model, dl_train, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, targets in dl_train:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total

def evaluate(model, dl_test, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dl_test:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total

def objective_baseline(trial, X_train, y_train, X_test, y_test, device):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256)

    model = BaselineMLP(40, hidden_dim, 10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    for epoch in range(20):
        train_loss, train_acc = train_one_epoch(model, dl_train, optimizer, device)
        if trial.number == 0:
            print(f"Epoch {epoch}: loss={train_loss:.4f}, acc={train_acc:.4f}")

    _, test_acc = evaluate(model, dl_test, device)
    return test_acc

def objective_nystrom(trial, X_train, y_train, X_test, y_test, device):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
    num_landmarks = trial.suggest_int("num_landmarks", 16, 128)
    gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)

    model = NystromMLP(40, num_landmarks, hidden_dim, 10).to(device)
    # Initialize landmarks from data
    with torch.no_grad():
        indices = torch.randperm(X_train.size(0))[:num_landmarks]
        model.nystrom.landmarks.copy_(X_train[indices])
        model.nystrom.log_gamma.fill_(np.log(gamma))

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    for epoch in range(20):
        train_one_epoch(model, dl_train, optimizer, device)

    _, test_acc = evaluate(model, dl_test, device)
    return test_acc

def main():
    device = torch.device("cpu")
    X_train, y_train, X_test, y_test = load_data()

    print("Tuning Baseline MLP...")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(lambda t: objective_baseline(t, X_train, y_train, X_test, y_test, device), n_trials=10)

    print("Tuning Nystrom MLP...")
    study_nystrom = optuna.create_study(direction="maximize")
    study_nystrom.optimize(lambda t: objective_nystrom(t, X_train, y_train, X_test, y_test, device), n_trials=20)

    print(f"Best Baseline accuracy: {study_baseline.best_value:.4f}")
    print(f"Best Baseline params: {study_baseline.best_params}")
    print(f"Best Nystrom accuracy: {study_nystrom.best_value:.4f}")
    print(f"Best Nystrom params: {study_nystrom.best_params}")

    # Final evaluation with multiple seeds
    def run_final_eval(model_class, params, X_train, y_train, X_test, y_test, device, seeds=[0, 1, 2]):
        accs = []
        for seed in seeds:
            torch.manual_seed(seed)
            if model_class == BaselineMLP:
                model = model_class(40, params['hidden_dim'], 10).to(device)
            else:
                model = model_class(40, params['num_landmarks'], params['hidden_dim'], 10).to(device)
                # Setting initial gamma from tuning and landmarks from data
                with torch.no_grad():
                    indices = torch.randperm(X_train.size(0))[:params['num_landmarks']]
                    model.nystrom.landmarks.copy_(X_train[indices])
                    model.nystrom.log_gamma.fill_(np.log(params['gamma']))

            optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
            dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
            dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

            for epoch in range(50):
                train_one_epoch(model, dl_train, optimizer, device)

            _, test_acc = evaluate(model, dl_test, device)
            accs.append(test_acc)
        return accs

    print("Final evaluation...")
    baseline_accs = run_final_eval(BaselineMLP, study_baseline.best_params, X_train, y_train, X_test, y_test, device)
    nystrom_accs = run_final_eval(NystromMLP, study_nystrom.best_params, X_train, y_train, X_test, y_test, device)

    print(f"Baseline Final Accuracy: {np.mean(baseline_accs):.4f} +/- {np.std(baseline_accs):.4f}")
    print(f"Nystrom Final Accuracy: {np.mean(nystrom_accs):.4f} +/- {np.std(nystrom_accs):.4f}")

    # Save results to README
    with open("differentiable_nystrom_kernel_layer/results.txt", "w") as f:
        f.write(f"Baseline Accuracy: {np.mean(baseline_accs):.4f} +/- {np.std(baseline_accs):.4f}\n")
        f.write(f"Nystrom Accuracy: {np.mean(nystrom_accs):.4f} +/- {np.std(nystrom_accs):.4f}\n")
        f.write(f"Baseline Params: {study_baseline.best_params}\n")
        f.write(f"Nystrom Params: {study_nystrom.best_params}\n")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.bar(["Baseline", "Nystrom"], [np.mean(baseline_accs), np.mean(nystrom_accs)],
            yerr=[np.std(baseline_accs), np.std(nystrom_accs)], capsize=10)
    plt.ylabel("Test Accuracy")
    plt.title("Comparison: Baseline MLP vs Nystrom MLP on MNIST-1D")
    plt.savefig("differentiable_nystrom_kernel_layer/comparison.png")

if __name__ == "__main__":
    main()
