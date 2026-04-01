import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
import optuna
import numpy as np
import matplotlib.pyplot as plt
from model import LVEN, BaselineMLP
import time
import os

def get_data():
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)
    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=50, batch_size=64):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    test_accs = []
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = F.cross_entropy(output, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Eval
        model.eval()
        with torch.no_grad():
            test_output = model(X_test)
            pred = test_output.argmax(dim=1)
            acc = (pred == y_test).float().mean().item()
            test_accs.append(acc)

    return test_accs

def objective_lven(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    X_train, y_train, X_test, y_test = get_data()
    model = LVEN(40, 10, latent_dim=32, num_experts=64)
    accs = train_model(model, X_train, y_train, X_test, y_test, lr, epochs=30)
    return max(accs)

def objective_mlp(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    X_train, y_train, X_test, y_test = get_data()
    model = BaselineMLP(40, 10, hidden_dim=170)
    accs = train_model(model, X_train, y_train, X_test, y_test, lr, epochs=30)
    return max(accs)

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    # Tuning
    print("Tuning LVEN...")
    study_lven = optuna.create_study(direction="maximize")
    study_lven.optimize(objective_lven, n_trials=8)
    best_lr_lven = study_lven.best_params["lr"]

    print("Tuning BaselineMLP...")
    study_mlp = optuna.create_study(direction="maximize")
    study_mlp.optimize(objective_mlp, n_trials=8)
    best_lr_mlp = study_mlp.best_params["lr"]

    print(f"Best LR LVEN: {best_lr_lven}")
    print(f"Best LR MLP: {best_lr_mlp}")

    # Final evaluation
    num_seeds = 3
    lven_results = []
    mlp_results = []

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        print(f"Seed {seed}: Training LVEN...")
        model_lven = LVEN(40, 10, latent_dim=32, num_experts=64)
        lven_accs = train_model(model_lven, X_train, y_train, X_test, y_test, best_lr_lven, epochs=50)
        lven_results.append(lven_accs)

        torch.manual_seed(seed)
        print(f"Seed {seed}: Training BaselineMLP...")
        model_mlp = BaselineMLP(40, 10, hidden_dim=170)
        mlp_accs = train_model(model_mlp, X_train, y_train, X_test, y_test, best_lr_mlp, epochs=50)
        mlp_results.append(mlp_accs)

    lven_mean = np.mean(lven_results, axis=0)
    lven_std = np.std(lven_results, axis=0)
    mlp_mean = np.mean(mlp_results, axis=0)
    mlp_std = np.std(mlp_results, axis=0)

    plt.figure(figsize=(10, 6))
    epochs = range(1, 51)
    plt.plot(epochs, lven_mean, label=f"LVEN (LR={best_lr_lven:.5f})", color="blue")
    plt.fill_between(epochs, lven_mean - lven_std, lven_mean + lven_std, color="blue", alpha=0.2)
    plt.plot(epochs, mlp_mean, label=f"BaselineMLP (LR={best_lr_mlp:.5f})", color="red")
    plt.fill_between(epochs, mlp_mean - mlp_std, mlp_mean + mlp_std, color="red", alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("LVEN vs BaselineMLP on MNIST-1D")
    plt.legend()
    plt.grid(True)
    plt.savefig("latent_voronoi_expert_network/comparison.png")

    with open("latent_voronoi_expert_network/results.txt", "w") as f:
        f.write(f"LVEN Final Accuracy: {lven_mean[-1]:.4f} +/- {lven_std[-1]:.4f}\n")
        f.write(f"BaselineMLP Final Accuracy: {mlp_mean[-1]:.4f} +/- {mlp_std[-1]:.4f}\n")
        f.write(f"Best LR LVEN: {best_lr_lven}\n")
        f.write(f"Best LR MLP: {best_lr_mlp}\n")

if __name__ == "__main__":
    run_experiment()
