import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from gradient_orthogonal_weight_decay.optimizer import GOWD
import numpy as np
import matplotlib.pyplot as plt
import os

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Dataset preparation
def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test'])
    return X_train, y_train, X_test, y_test

# Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.net(x)

def train_and_eval(lr, wd, opt_type, X_train, y_train, X_test, y_test, epochs=30, return_curve=False):
    model = MLP()
    if opt_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = GOWD(model.parameters(), lr=lr, weight_decay=wd)

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc = evaluate(model, X_test, y_test)
        history.append(acc)

    if return_curve:
        return history
    return history[-1]

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        preds = outputs.argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
    return acc

def objective(trial, X_train, y_train, X_test, y_test, opt_type):
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    return train_and_eval(lr, wd, opt_type, X_train, y_train, X_test, y_test, epochs=20)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    print("Running AdamW trials...")
    study_adamw = optuna.create_study(direction="maximize")
    study_adamw.optimize(lambda t: objective(t, X_train, y_train, X_test, y_test, "AdamW"), n_trials=30)

    print("Running GOWD trials...")
    study_gowd = optuna.create_study(direction="maximize")
    study_gowd.optimize(lambda t: objective(t, X_train, y_train, X_test, y_test, "GOWD"), n_trials=30)

    print(f"Best AdamW Params: {study_adamw.best_params}")
    print(f"Best AdamW Accuracy: {study_adamw.best_value:.4f}")
    print(f"Best GOWD Params: {study_gowd.best_params}")
    print(f"Best GOWD Accuracy: {study_gowd.best_value:.4f}")

    # Final run with best params to get curves
    print("Running final comparison with best hyperparameters...")
    adamw_history = train_and_eval(study_adamw.best_params['lr'], study_adamw.best_params['weight_decay'],
                                   "AdamW", X_train, y_train, X_test, y_test, epochs=50, return_curve=True)

    gowd_history = train_and_eval(study_gowd.best_params['lr'], study_gowd.best_params['weight_decay'],
                                  "GOWD", X_train, y_train, X_test, y_test, epochs=50, return_curve=True)

    plt.figure(figsize=(10, 6))
    plt.plot(adamw_history, label=f"AdamW (acc={study_adamw.best_value:.4f})")
    plt.plot(gowd_history, label=f"GOWD (acc={study_gowd.best_value:.4f})")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("AdamW vs GOWD on MNIST1D")
    plt.legend()
    plt.grid(True)
    plt.savefig("gradient_orthogonal_weight_decay/results.png")

    with open("gradient_orthogonal_weight_decay/results.txt", "w") as f:
        f.write(f"AdamW Best Accuracy: {study_adamw.best_value:.4f}\n")
        f.write(f"AdamW Best Params: {study_adamw.best_params}\n")
        f.write(f"GOWD Best Accuracy: {study_gowd.best_value:.4f}\n")
        f.write(f"GOWD Best Params: {study_gowd.best_params}\n")
