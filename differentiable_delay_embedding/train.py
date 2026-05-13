import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import DLDEModel, BaselineMLP
import os

def get_data():
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_model(model, train_loader, test_loader, lr, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total

def objective(trial, model_type):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    X_train, y_train, X_test, y_test = get_data()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    if model_type == "DLDE":
        model = DLDEModel()
    else:
        model = BaselineMLP()

    accuracy = train_model(model, train_loader, test_loader, lr, epochs=20)
    return accuracy

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    print("Tuning DLDE Model...")
    study_dlde = optuna.create_study(direction="maximize")
    study_dlde.optimize(lambda trial: objective(trial, "DLDE"), n_trials=10)
    best_lr_dlde = study_dlde.best_params["lr"]

    print("Tuning Baseline Model...")
    study_base = optuna.create_study(direction="maximize")
    study_base.optimize(lambda trial: objective(trial, "Baseline"), n_trials=10)
    best_lr_base = study_base.best_params["lr"]

    print(f"Best LR DLDE: {best_lr_dlde}")
    print(f"Best LR Baseline: {best_lr_base}")

    # Final training
    dlde_accs = []
    base_accs = []
    learned_taus = []

    for i in range(5):
        print(f"Run {i+1}/5")
        model_dlde = DLDEModel()
        acc_dlde = train_model(model_dlde, train_loader, test_loader, best_lr_dlde, epochs=50)
        dlde_accs.append(acc_dlde)
        learned_taus.append(model_dlde.dlde.tau.item())

        model_base = BaselineMLP()
        acc_base = train_model(model_base, train_loader, test_loader, best_lr_base, epochs=50)
        base_accs.append(acc_base)

    results = f"""
DLDE Accuracy: {np.mean(dlde_accs):.4f} +/- {np.std(dlde_accs):.4f}
Baseline Accuracy: {np.mean(base_accs):.4f} +/- {np.std(base_accs):.4f}
Learned Taus: {learned_taus}
Mean Learned Tau: {np.mean(learned_taus):.4f}
"""
    print(results)
    with open("differentiable_delay_embedding/results.txt", "w") as f:
        f.write(results)

    plt.figure()
    plt.hist(learned_taus)
    plt.title("Distribution of Learned Taus")
    plt.xlabel("Tau")
    plt.ylabel("Frequency")
    plt.savefig("differentiable_delay_embedding/learned_taus.png")

    # Create README.md
    with open("differentiable_delay_embedding/README.md", "w") as f:
        f.write(f"# Differentiable Learnable Delay Embedding Experiment\n\n")
        f.write("This experiment investigates making the delay parameter $\\tau$ in delay embedding a learnable parameter using linear interpolation.\n\n")
        f.write("## Results\n")
        f.write(results.replace("\n", "\n\n"))
        f.write("\n## Visualization\n")
        f.write("![Learned Taus](learned_taus.png)\n")
