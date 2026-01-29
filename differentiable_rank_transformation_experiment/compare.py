import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna
import matplotlib.pyplot as plt
import os
from mnist1d.data import get_dataset_args, make_dataset
from model import TabularMLP

def get_data():
    args = get_dataset_args()
    data = make_dataset(args)
    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def apply_distortions(X):
    # Apply some non-linear monotonic distortions to features
    # This simulates real-world tabular data where features might be skewed or have outliers
    X_distorted = X.clone()
    num_features = X.shape[1]

    for i in range(num_features):
        if i % 4 == 0:
            # Power transform
            X_distorted[:, i] = torch.pow(torch.abs(X[:, i]), 1.5) * torch.sign(X[:, i])
        elif i % 4 == 1:
            # Exponential-like transform
            X_distorted[:, i] = torch.expm1(torch.abs(X[:, i])) * torch.sign(X[:, i])
        elif i % 4 == 2:
            # Log-like transform
            X_distorted[:, i] = torch.log1p(torch.abs(X[:, i])) * torch.sign(X[:, i])
        # i % 4 == 3 stays linear
    return X_distorted

def train_and_evaluate(config, X_train, y_train, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = TabularMLP(
        input_dim=X_train.shape[1],
        hidden_dim=config['hidden_dim'],
        output_dim=10,
        num_layers=config['num_layers'],
        use_dbrt=config['use_dbrt'],
        dbrt_only=config['dbrt_only']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(config['epochs']):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_acc = 100 * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def objective(trial, X_train, y_train, X_test, y_test, use_dbrt, dbrt_only):
    config = {
        'hidden_dim': trial.suggest_int('hidden_dim', 32, 256),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'epochs': 25,
        'use_dbrt': use_dbrt,
        'dbrt_only': dbrt_only
    }
    return train_and_evaluate(config, X_train, y_train, X_test, y_test)

def run_experiment(name, X_train, y_train, X_test, y_test, use_dbrt, dbrt_only):
    print(f"Running experiment: {name}")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, use_dbrt, dbrt_only), n_trials=10)
    print(f"Best accuracy for {name}: {study.best_value:.2f}%")
    return study.best_value, study.best_params

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    # We will run experiments on both original and distorted data
    X_train_dist = apply_distortions(X_train)
    X_test_dist = apply_distortions(X_test)

    results = {}

    # 1. Baseline MLP on Original Data
    acc, params = run_experiment("Baseline_Original", X_train, y_train, X_test, y_test, False, False)
    results["Baseline_Original"] = acc

    # 2. MLP + DBRT on Original Data
    acc, params = run_experiment("DBRT_Original", X_train, y_train, X_test, y_test, True, False)
    results["DBRT_Original"] = acc

    # 3. DBRT Only on Original Data
    acc, params = run_experiment("DBRT_Only_Original", X_train, y_train, X_test, y_test, True, True)
    results["DBRT_Only_Original"] = acc

    # 4. Baseline MLP on Distorted Data
    acc, params = run_experiment("Baseline_Distorted", X_train_dist, y_train, X_test_dist, y_test, False, False)
    results["Baseline_Distorted"] = acc

    # 5. MLP + DBRT on Distorted Data
    acc, params = run_experiment("DBRT_Distorted", X_train_dist, y_train, X_test_dist, y_test, True, False)
    results["DBRT_Distorted"] = acc

    # 6. DBRT-Only on Distorted Data
    acc, params = run_experiment("DBRT_Only_Distorted", X_train_dist, y_train, X_test_dist, y_test, True, True)
    results["DBRT_Only_Distorted"] = acc

    print("\nSummary of Results:")
    for name, acc in results.items():
        print(f"{name}: {acc:.2f}%")

    with open(os.path.join(os.path.dirname(__file__), "results.txt"), "w") as f:
        for name, acc in results.items():
            f.write(f"{name}: {acc:.2f}%\n")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(results.keys(), results.values())
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Differentiable Batch Rank Transformation Comparison')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "results.png"))
