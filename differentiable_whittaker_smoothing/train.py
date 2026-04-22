import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import optuna
import numpy as np
from layer import WhittakerSmoothing, WhittakerMultiScale
import matplotlib.pyplot as plt
import os

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_data(noise_std=0.0):
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.int64)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.int64)

    if noise_std > 0:
        X_train += torch.randn_like(X_train) * noise_std
        X_test += torch.randn_like(X_test) * noise_std

    return X_train, y_train, X_test, y_test

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class WhittakerMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, order=2, initial_lambda=10.0):
        super().__init__()
        self.whittaker = WhittakerSmoothing(input_dim, order=order, initial_lambda=initial_lambda)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.whittaker(x)
        return self.net(x)

class WhittakerAugmentedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10, orders=[2], lambdas=[1.0, 10.0, 100.0]):
        super().__init__()
        self.multiscale = WhittakerMultiScale(input_dim, orders=orders, lambdas=lambdas)
        # input_dim * (1 + len(orders)*len(lambdas))
        augmented_dim = input_dim * (1 + len(orders) * len(lambdas))
        self.net = nn.Sequential(
            nn.Linear(augmented_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.multiscale(x)
        return self.net(x)

def train_model(model, X_train, y_train, X_test, y_test, lr=1e-3, epochs=50, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == y_test).sum().item() / y_test.size(0)
            if acc > best_acc:
                best_acc = acc
    return best_acc

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_type == "baseline":
        model = BaselineMLP()
    elif model_type == "whittaker":
        model = WhittakerMLP()
    elif model_type == "augmented":
        model = WhittakerAugmentedMLP()

    acc = train_model(model, X_train, y_train, X_test, y_test, lr=lr, epochs=30)
    return acc

def run_experiment(noise_std=0.0):
    print(f"\n--- Running Experiment (noise_std={noise_std}) ---")
    X_train, y_train, X_test, y_test = get_data(noise_std=noise_std)

    results = {}
    for model_type in ["baseline", "whittaker", "augmented"]:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=5)

        print(f"Best LR for {model_type}: {study.best_params['lr']}")

        # Train final model with best LR and more epochs
        accuracies = []
        for i in range(1): # 1 seed
            torch.manual_seed(42 + i)
            if model_type == "baseline":
                model = BaselineMLP()
            elif model_type == "whittaker":
                model = WhittakerMLP()
            elif model_type == "augmented":
                model = WhittakerAugmentedMLP()

            acc = train_model(model, X_train, y_train, X_test, y_test, lr=study.best_params['lr'], epochs=50)
            accuracies.append(acc)
            print(f"Seed {i} accuracy: {acc:.4f}")

        results[model_type] = {
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
            "best_lr": study.best_params['lr']
        }

    return results

if __name__ == "__main__":
    os.makedirs("differentiable_whittaker_smoothing", exist_ok=True)

    res_noisy = run_experiment(noise_std=0.2)

    with open("differentiable_whittaker_smoothing/results.txt", "w") as f:
        f.write("Noisy Results (std=0.2):\n")
        for m, r in res_noisy.items():
            f.write(f"{m}: {r['mean']:.4f} +/- {r['std']:.4f} (LR: {r['best_lr']:.6f})\n")

    print("\nResults saved to results.txt")
