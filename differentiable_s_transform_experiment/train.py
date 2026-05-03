import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from model import STAugmentedMLP, STMLP, BaselineMLP
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=50, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in dl_train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test, y_test = X_test.to(device), y_test.to(device)
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    return accuracy

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_type == "Baseline":
        model = BaselineMLP(hidden_dim=256)
    elif model_type == "BaselineWide":
        model = BaselineMLP(hidden_dim=673)
    elif model_type == "STAug":
        model = STAugmentedMLP(hidden_dim=256)
    elif model_type == "ST":
        model = STMLP(hidden_dim=256)

    return train_model(model, X_train, y_train, X_test, y_test, lr, epochs=20)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    results = {}

    for model_type in ["Baseline", "BaselineWide", "STAug", "ST"]:
        print(f"Optimizing {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=5)

        best_lr = study.best_params["lr"]
        print(f"Best LR for {model_type}: {best_lr}")

        # Run 5 seeds with best LR
        accs = []
        for seed in range(5):
            torch.manual_seed(seed)
            if model_type == "Baseline":
                model = BaselineMLP(hidden_dim=256)
            elif model_type == "BaselineWide":
                model = BaselineMLP(hidden_dim=673)
            elif model_type == "STAug":
                model = STAugmentedMLP(hidden_dim=256)
            elif model_type == "ST":
                model = STMLP(hidden_dim=256)

            acc = train_model(model, X_train, y_train, X_test, y_test, best_lr)
            accs.append(acc)
            print(f"Seed {seed} accuracy: {acc:.4f}")

        results[model_type] = (np.mean(accs), np.std(accs))
        print(f"{model_type} final results: {results[model_type][0]:.4f} +/- {results[model_type][1]:.4f}")

    with open("differentiable_s_transform_experiment/results.txt", "w") as f:
        for model_type, (mean, std) in results.items():
            f.write(f"{model_type}: {mean:.4f} +/- {std:.4f}\n")
