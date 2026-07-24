import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from differentiable_lof_experiment.model import BaselineMLP, LOFRegularizedMLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, lr, epochs=30, lof_weight=0.0):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            if isinstance(model, LOFRegularizedMLP) and lof_weight > 0:
                outputs, lof_loss = model(inputs, return_lof=True)
                loss = criterion(outputs, targets) + lof_weight * lof_loss
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).float().mean().item()
    return accuracy

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lof_weight = 0.0
    if model_type == "lof":
        lof_weight = trial.suggest_float("lof_weight", 1e-4, 1.0, log=True)
        model = LOFRegularizedMLP()
    else:
        model = BaselineMLP()

    # Simple split for validation
    train_size = int(0.8 * len(X_train))
    X_t, y_t = X_train[:train_size], y_train[:train_size]
    X_v, y_v = X_train[train_size:], y_train[train_size:]

    train_model(model, X_t, y_t, lr, epochs=15, lof_weight=lof_weight)
    return evaluate_model(model, X_v, y_v)

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    # Tuning Baseline
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(lambda trial: objective(trial, "baseline", X_train, y_train, X_test, y_test), n_trials=20)
    best_lr_baseline = study_baseline.best_params["lr"]

    # Tuning LOF
    study_lof = optuna.create_study(direction="maximize")
    study_lof.optimize(lambda trial: objective(trial, "lof", X_train, y_train, X_test, y_test), n_trials=20)
    best_lr_lof = study_lof.best_params["lr"]
    best_lof_weight = study_lof.best_params["lof_weight"]

    print(f"Best Baseline LR: {best_lr_baseline}")
    print(f"Best LOF LR: {best_lr_lof}, Weight: {best_lof_weight}")

    # Final Evaluation
    seeds = [42, 43, 44, 45, 46]
    results = {"baseline": [], "lof": []}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Baseline
        model_b = BaselineMLP()
        train_model(model_b, X_train, y_train, best_lr_baseline, epochs=30)
        results["baseline"].append(evaluate_model(model_b, X_test, y_test))

        # LOF
        model_l = LOFRegularizedMLP()
        train_model(model_l, X_train, y_train, best_lr_lof, epochs=30, lof_weight=best_lof_weight)
        results["lof"].append(evaluate_model(model_l, X_test, y_test))

    for k, v in results.items():
        print(f"{k}: {np.mean(v):.4f} +/- {np.std(v):.4f}")

    with open("differentiable_lof_experiment/results.txt", "w") as f:
        f.write(f"Baseline: {np.mean(results['baseline']):.4f} +/- {np.std(results['baseline']):.4f}\n")
        f.write(f"LOF: {np.mean(results['lof']):.4f} +/- {np.std(results['lof']):.4f}\n")
        f.write(f"Best Baseline LR: {best_lr_baseline}\n")
        f.write(f"Best LOF LR: {best_lr_lof}, Weight: {best_lof_weight}\n")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.boxplot([results["baseline"], results["lof"]])
    plt.xticks([1, 2], ["Baseline", "LOF Regularized"])
    plt.title("MNIST-1D Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.savefig("differentiable_lof_experiment/results.png")

if __name__ == "__main__":
    run_experiment()
