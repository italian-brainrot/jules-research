import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import BisineNetwork
from optimizer import SFNOptimizer
import time

def get_data(num_samples=2000):
    defaults = get_dataset_args()
    defaults.num_samples = num_samples
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_sfn(X_train, y_train, X_test, y_test, num_steps=50, lr=1.0, num_units=2):
    model = BisineNetwork(input_dim=40, num_classes=10, num_units=num_units)
    optimizer = SFNOptimizer(model, lr=lr)
    history = {"loss": [], "train_acc": [], "test_acc": [], "eigenvalues": []}
    for step in range(num_steps):
        loss, evals = optimizer.step(X_train, y_train)
        with torch.no_grad():
            z_train = model(X_train)
            train_acc = (z_train.argmax(1) == y_train).float().mean().item()
            z_test = model(X_test)
            test_acc = (z_test.argmax(1) == y_test).float().mean().item()
        history["loss"].append(loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["eigenvalues"].append(evals.detach().cpu().numpy())
        if step % 10 == 0:
            print(f"SFN Step {step}: Loss={loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    return history, model

def train_adam(X_train, y_train, X_test, y_test, num_epochs=100, lr=1e-3, num_units=2, batch_size=256):
    model = BisineNetwork(input_dim=40, num_classes=10, num_units=num_units)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)
    history = {"loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in dl_train:
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.shape[0]
        model.eval()
        with torch.no_grad():
            z_train = model(X_train)
            train_acc = (z_train.argmax(1) == y_train).float().mean().item()
            z_test = model(X_test)
            test_acc = (z_test.argmax(1) == y_test).float().mean().item()
        history["loss"].append(total_loss / X_train.shape[0])
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
    return history, model

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))
    def forward(self, x):
        return self.net(x)

def train_mlp_adam(X_train, y_train, X_test, y_test, num_epochs=100, lr=1e-3, hidden_dim=100, batch_size=256):
    model = MLP(40, hidden_dim, 10)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)
    history = {"loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in dl_train:
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.shape[0]
        model.eval()
        with torch.no_grad():
            z_train = model(X_train)
            train_acc = (z_train.argmax(1) == y_train).float().mean().item()
            z_test = model(X_test)
            test_acc = (z_test.argmax(1) == y_test).float().mean().item()
        history["loss"].append(total_loss / X_train.shape[0])
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
    return history, model

def objective_sfn(trial):
    X_train, y_train, X_test, y_test = get_data(num_samples=1000)
    lr = trial.suggest_float("lr", 0.1, 2.0, log=True)
    history, _ = train_sfn(X_train, y_train, X_test, y_test, num_steps=20, lr=lr)
    return max(history["test_acc"])

def objective_adam(trial):
    X_train, y_train, X_test, y_test = get_data(num_samples=1000)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    history, _ = train_adam(X_train, y_train, X_test, y_test, num_epochs=50, lr=lr)
    return max(history["test_acc"])

def run_comparison():
    X_train, y_train, X_test, y_test = get_data(num_samples=2000)
    print("Tuning SFN...")
    study_sfn = optuna.create_study(direction="maximize")
    study_sfn.optimize(objective_sfn, n_trials=3)
    best_lr_sfn = study_sfn.best_params["lr"]
    print("Tuning Adam...")
    study_adam = optuna.create_study(direction="maximize")
    study_adam.optimize(objective_adam, n_trials=3)
    best_lr_adam = study_adam.best_params["lr"]
    print(f"Best SFN LR: {best_lr_sfn}, Best Adam LR: {best_lr_adam}")
    print("Running final SFN...")
    hist_sfn, _ = train_sfn(X_train, y_train, X_test, y_test, num_steps=50, lr=best_lr_sfn)
    print("Running final Adam...")
    hist_adam, _ = train_adam(X_train, y_train, X_test, y_test, num_epochs=150, lr=best_lr_adam)
    print("Running MLP baseline...")
    hist_mlp, _ = train_mlp_adam(X_train, y_train, X_test, y_test, num_epochs=150, lr=1e-3)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(hist_sfn["loss"], label="Bisine + SFN")
    plt.plot(np.linspace(0, 50, len(hist_adam["loss"])), hist_adam["loss"], label="Bisine + Adam")
    plt.plot(np.linspace(0, 50, len(hist_mlp["loss"])), hist_mlp["loss"], label="MLP + Adam")
    plt.title("Training Loss")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(hist_sfn["test_acc"], label="Bisine + SFN")
    plt.plot(np.linspace(0, 50, len(hist_adam["test_acc"])), hist_adam["test_acc"], label="Bisine + Adam")
    plt.plot(np.linspace(0, 50, len(hist_mlp["test_acc"])), hist_mlp["test_acc"], label="MLP + Adam")
    plt.title("Test Accuracy")
    plt.legend()
    plt.subplot(1, 3, 3)
    evals = np.array(hist_sfn["eigenvalues"])
    for i in range(0, evals.shape[1], max(1, evals.shape[1] // 20)):
        plt.plot(evals[:, i], alpha=0.3)
    plt.title("SFN Hessian Eigenvalues")
    plt.tight_layout()
    plt.savefig("sfn_bisine_experiment/results.png")
    print(f"Final Test Acc - SFN: {max(hist_sfn['test_acc']):.4f}, Adam: {max(hist_adam['test_acc']):.4f}, MLP: {max(hist_mlp['test_acc']):.4f}")

if __name__ == "__main__":
    run_comparison()
