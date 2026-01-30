import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import TRLModel, MLPModel

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 4000 # Using default to save time
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data['x_test']).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        out = model(X)
        preds = out.argmax(dim=1)
        acc = (preds == y).float().mean().item()
        loss = nn.CrossEntropyLoss()(out, y).item()
    return acc, loss

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    if model_type == "TRL":
        model = TRLModel()
    else:
        model = MLPModel()

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Quick training for tuning
    for epoch in range(20):
        model.train()
        for xb, yb in dl_train:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    acc, _ = evaluate(model, X_test, y_test)
    return acc

def main():
    X_train, y_train, X_test, y_test = get_data()

    print("Tuning TRL Model...")
    study_trl = optuna.create_study(direction="maximize")
    study_trl.optimize(lambda t: objective(t, "TRL", X_train, y_train, X_test, y_test), n_trials=10)
    best_lr_trl = study_trl.best_params["lr"]
    print(f"Best LR for TRL: {best_lr_trl}")

    print("Tuning MLP Model...")
    study_mlp = optuna.create_study(direction="maximize")
    study_mlp.optimize(lambda t: objective(t, "MLP", X_train, y_train, X_test, y_test), n_trials=10)
    best_lr_mlp = study_mlp.best_params["lr"]
    print(f"Best LR for MLP: {best_lr_mlp}")

    # Train both with best LRs
    trl_model = TRLModel()
    mlp_model = MLPModel()

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

    history_trl = []
    history_mlp = []

    print("\nTraining TRL Model...")
    optimizer_trl = optim.Adam(trl_model.parameters(), lr=best_lr_trl)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(50):
        trl_model.train()
        for xb, yb in dl_train:
            optimizer_trl.zero_grad()
            out = trl_model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer_trl.step()
        acc, _ = evaluate(trl_model, X_test, y_test)
        history_trl.append(acc)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} TRL Test Acc: {acc:.4f}")

    print("\nTraining MLP Model...")
    optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=best_lr_mlp)
    for epoch in range(50):
        mlp_model.train()
        for xb, yb in dl_train:
            optimizer_mlp.zero_grad()
            out = mlp_model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer_mlp.step()
        acc, _ = evaluate(mlp_model, X_test, y_test)
        history_mlp.append(acc)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} MLP Test Acc: {acc:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(history_trl, label=f"TRL-Model (lr={best_lr_trl:.2e})")
    plt.plot(history_mlp, label=f"MLP-Model (lr={best_lr_mlp:.2e})")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.title("Comparison: TRL-Model vs MLP-Model on mnist1d")
    plt.grid(True)
    plt.savefig("tridiagonal_resolvent_experiment/comparison.png")

    # Visualization of M^-1
    trl_model.eval()
    with torch.no_grad():
        diag = F.softplus(trl_model.trl1.d) + 2.0
        off_l = torch.tanh(trl_model.trl1.l)
        off_u = torch.tanh(trl_model.trl1.u)
        M = torch.diag(diag) + torch.diag(off_l, -1) + torch.diag(off_u, 1)
        Minv = torch.inverse(M).cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(Minv, cmap='viridis')
    plt.colorbar()
    plt.title("Learned M^-1 (TRL Layer 1)")
    plt.savefig("tridiagonal_resolvent_experiment/Minv_layer1.png")

    print("\nExperiment complete. Results saved in tridiagonal_resolvent_experiment/")

if __name__ == "__main__":
    main()
