import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import AdaptiveMLP
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_eval(params, X_train, y_train, X_test, y_test, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptiveMLP(
        hidden_dim=256,
        p_base=params['p_base'],
        gamma=params['gamma']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    dl_train = TensorDataLoader((X_train.to(device), y_train.to(device)), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test.to(device), y_test.to(device)), batch_size=128, shuffle=False)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dl_test:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total

def objective(trial):
    X_train, y_train, X_test, y_test = get_data()

    params = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'p_base': trial.suggest_float('p_base', 0.0, 0.7),
        'gamma': trial.suggest_float('gamma', -2.0, 2.0)
    }

    return train_eval(params, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best trials:")
    trials = study.trials
    # Filter trials by gamma to find best in different regimes

    baseline_trials = [t for t in trials if abs(t.params['gamma']) < 0.1]
    ftd_trials = [t for t in trials if t.params['gamma'] > 0.5]
    fsd_trials = [t for t in trials if t.params['gamma'] < -0.5]

    def print_best(name, filtered_trials):
        if not filtered_trials:
            print(f"No trials for {name}")
            return
        best = max(filtered_trials, key=lambda t: t.value)
        print(f"{name}: Best Value {best.value}, Params {best.params}")

    print_best("Baseline (near gamma=0)", baseline_trials)
    print_best("FTD (gamma > 0.5)", ftd_trials)
    print_best("FSD (gamma < -0.5)", fsd_trials)

    best_overall = study.best_trial
    print(f"Overall Best: Value {best_overall.value}, Params {best_overall.params}")

    # Save best params for compare.py
    import json
    with open('fisher_adaptive_dropout_experiment/best_params.json', 'w') as f:
        json.dump(study.best_params, f)
