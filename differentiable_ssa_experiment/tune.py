import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from ssa import SSANet

class BaselineMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def get_noisy_mnist1d(noise_std=0.2):
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)

    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y'])
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test'])

    # Add noise to training and test sets
    X_train += noise_std * torch.randn_like(X_train)
    X_test += noise_std * torch.randn_like(X_test)

    return X_train, y_train, X_test, y_test

def train_model(model, dl_train, dl_test, lr, epochs=40):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in dl_train:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl_test:
            out = model(x)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return 100 * correct / total

def objective_baseline(trial):
    X_train, y_train, X_test, y_test = get_noisy_mnist1d()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    hidden_size = 128

    model = BaselineMLP(40, hidden_size, 10)
    acc = train_model(model, dl_train, dl_test, lr)
    return acc

def objective_ssa(trial):
    X_train, y_train, X_test, y_test = get_noisy_mnist1d()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    window_size = trial.suggest_int("window_size", 4, 15)
    hidden_size = 128

    model = SSANet(40, window_size, hidden_size, 10)
    acc = train_model(model, dl_train, dl_test, lr)
    return acc

if __name__ == "__main__":
    print("Tuning Baseline...")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(objective_baseline, n_trials=15)

    print("Tuning SSA...")
    study_ssa = optuna.create_study(direction="maximize")
    study_ssa.optimize(objective_ssa, n_trials=15)

    print("\nBest Baseline Params:", study_baseline.best_params)
    print("Best Baseline Acc:", study_baseline.best_value)
    print("\nBest SSA Params:", study_ssa.best_params)
    print("Best SSA Acc:", study_ssa.best_value)
