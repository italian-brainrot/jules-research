import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from log_linear_interaction_network.model import BaselineMLP, LLIN
import numpy as np
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

def train_eval(model, train_loader, test_loader, lr, weight_decay, epochs=20):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        acc = correct / total
        if acc > best_acc:
            best_acc = acc

    return best_acc

def objective(trial, model_type):
    X_train, y_train, X_test, y_test = get_data()

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    batch_size = trial.suggest_categorical("batch_size", [128, 256])

    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=batch_size, shuffle=False)

    if model_type == "baseline":
        model = BaselineMLP(40, 10, hidden_dim, n_layers)
    else:
        model = LLIN(40, 10, hidden_dim, n_layers)

    return train_eval(model, train_loader, test_loader, lr, weight_decay)

if __name__ == "__main__":
    for m_type in ["llin", "baseline"]:
        if os.path.exists(f"log_linear_interaction_network/best_params_{m_type}.txt"):
            print(f"Skipping {m_type}, already tuned.")
            continue

        print(f"Tuning {m_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, m_type), n_trials=10)

        print(f"Best trial for {m_type}:")
        print(f"  Value: {study.best_trial.value}")
        print(f"  Params: {study.best_trial.params}")

        # Save best params
        with open(f"log_linear_interaction_network/best_params_{m_type}.txt", "w") as f:
            f.write(str(study.best_trial.params))
