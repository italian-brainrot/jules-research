import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import BaselineMLP, DZTAugmentedMLP, DZTNet
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=50, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    return accuracy

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    input_dim = 40
    output_dim = 10
    hidden_dim = 128
    num_points = 20

    if model_type == "baseline":
        model = BaselineMLP(input_dim, hidden_dim, output_dim)
    elif model_type == "dzt_augmented":
        model = DZTAugmentedMLP(input_dim, num_points, hidden_dim, output_dim)
    else:
        model = DZTNet(input_dim, num_points, hidden_dim, output_dim)

    return train_model(model, X_train, y_train, X_test, y_test, lr, epochs=30)

def main():
    X_train, y_train, X_test, y_test = get_data()

    best_lrs = {}
    for model_type in ["baseline", "dzt_augmented", "dzt_only"]:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=15)
        best_lrs[model_type] = study.best_params["lr"]
        print(f"Best LR for {model_type}: {best_lrs[model_type]}")

    results = {}
    num_seeds = 3
    for model_type in ["baseline", "dzt_augmented", "dzt_only"]:
        accs = []
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            input_dim = 40
            output_dim = 10
            hidden_dim = 128
            num_points = 20
            if model_type == "baseline":
                model = BaselineMLP(input_dim, hidden_dim, output_dim)
            elif model_type == "dzt_augmented":
                model = DZTAugmentedMLP(input_dim, num_points, hidden_dim, output_dim)
            else:
                model = DZTNet(input_dim, num_points, hidden_dim, output_dim)

            acc = train_model(model, X_train, y_train, X_test, y_test, best_lrs[model_type], epochs=100)
            accs.append(acc)

            if model_type != "baseline":
                # Save learned points for one seed
                if seed == 0:
                    gamma = model.dzt.gamma.detach().cpu().numpy()
                    omega = model.dzt.omega.detach().cpu().numpy()

                    z_real = np.exp(gamma) * np.cos(omega)
                    z_imag = np.exp(gamma) * np.sin(omega)

                    plt.figure(figsize=(6,6))
                    theta = np.linspace(0, 2*np.pi, 100)
                    plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
                    plt.scatter(z_real, z_imag, c='r', label='Learned z_k')
                    plt.axhline(0, color='black', lw=1)
                    plt.axvline(0, color='black', lw=1)
                    plt.title(f"Learned Z-points for {model_type}")
                    plt.xlabel("Real")
                    plt.ylabel("Imag")
                    plt.grid(True)
                    plt.savefig(f"differentiable_z_transform_experiment/z_points_{model_type}.png")
                    plt.close()

        results[model_type] = (np.mean(accs), np.std(accs))
        print(f"{model_type}: {results[model_type][0]:.4f} +/- {results[model_type][1]:.4f}")

    with open("differentiable_z_transform_experiment/results.txt", "w") as f:
        for model_type, (mean, std) in results.items():
            f.write(f"{model_type}: {mean:.4f} +/- {std:.4f} (LR: {best_lrs[model_type]:.6f})\n")

if __name__ == "__main__":
    main()
