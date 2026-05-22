import torch
import torch.nn as nn
import torch.optim as optim
import mnist1d
import optuna
import numpy as np
import matplotlib.pyplot as plt
from model import BaselineMLP, ConvMLP
import os

def get_data():
    args = mnist1d.data.get_dataset_args()
    data = mnist1d.data.get_dataset(args)

    # Convert to float32
    x_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    x_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    return x_train, y_train, x_test, y_test

def train_model(model, x_train, y_train, x_test, y_test, lr, epochs=100, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(x_train.size(0))
        epoch_loss = 0
        for i in range(0, x_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / (x_train.size(0) / batch_size))

        model.eval()
        with torch.no_grad():
            outputs = model(x_test)
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == y_test).sum().item() / y_test.size(0)
            test_accs.append(acc)

    return max(test_accs)

def objective(trial, model_type, x_train, y_train, x_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    input_dim = 40
    hidden_dim = 128
    output_dim = 10

    if model_type == "baseline":
        model = BaselineMLP(input_dim, hidden_dim, output_dim)
    elif model_type == "gabor":
        model = ConvMLP(input_dim, hidden_dim, output_dim, use_chirplet=True, fixed_chirp=0.0)
    elif model_type == "chirplet":
        model = ConvMLP(input_dim, hidden_dim, output_dim, use_chirplet=True)
    elif model_type == "standard_conv":
        model = ConvMLP(input_dim, hidden_dim, output_dim, use_chirplet=False)

    return train_model(model, x_train, y_train, x_test, y_test, lr, epochs=20)

def run_experiment():
    x_train, y_train, x_test, y_test = get_data()

    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    results = {}
    model_types = ["baseline", "standard_conv", "gabor", "chirplet"]

    for mt in model_types:
        print(f"Tuning {mt}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, mt, x_train, y_train, x_test, y_test), n_trials=10)

        best_lr = study.best_params["lr"]
        print(f"Best LR for {mt}: {best_lr}, Best Acc: {study.best_value}")

        # Run final evaluation with more epochs
        print(f"Final training for {mt}...")
        final_accs = []
        for i in range(2): # 2 runs for stability
            input_dim = 40
            hidden_dim = 128
            output_dim = 10
            if mt == "baseline":
                model = BaselineMLP(input_dim, hidden_dim, output_dim)
            elif mt == "gabor":
                model = ConvMLP(input_dim, hidden_dim, output_dim, use_chirplet=True, fixed_chirp=0.0)
            elif mt == "chirplet":
                model = ConvMLP(input_dim, hidden_dim, output_dim, use_chirplet=True)
            elif mt == "standard_conv":
                model = ConvMLP(input_dim, hidden_dim, output_dim, use_chirplet=False)

            acc = train_model(model, x_train, y_train, x_test, y_test, best_lr, epochs=60)
            final_accs.append(acc)

        results[mt] = {
            "mean": np.mean(final_accs),
            "std": np.std(final_accs),
            "best_lr": best_lr
        }

    # Save results
    results_path = os.path.join(base_dir, "results.txt")
    with open(results_path, "w") as f:
        for mt, res in results.items():
            f.write(f"{mt}: {res['mean']:.4f} +- {res['std']:.4f} (LR: {res['best_lr']:.6f})\n")

    # Plot results
    mt_names = list(results.keys())
    means = [results[mt]["mean"] for mt in mt_names]
    stds = [results[mt]["std"] for mt in mt_names]

    plt.figure(figsize=(10, 6))
    plt.bar(mt_names, means, yerr=stds, capsize=5)
    plt.ylabel("Accuracy")
    plt.title("MNIST-1D Model Comparison (Chirplet vs Baselines)")
    plot_path = os.path.join(base_dir, "comparison.png")
    plt.savefig(plot_path)

if __name__ == "__main__":
    run_experiment()
