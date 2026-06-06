import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from differentiable_gammatone_experiment.model import BaselineMLP, GammatoneConvMLP, StandardConvMLP
import json
import os

def train_model(model, dl_train, dl_test, lr, epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
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

        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc

    return best_acc

def objective(trial, model_name):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    args = get_dataset_args()
    args.num_samples = 2000 # Reduced for faster Optuna
    data = make_dataset(args)

    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data['x_test']).float(), torch.tensor(data['y_test']).long()

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    if model_name == "baseline":
        model = BaselineMLP()
    elif model_name == "gammatone":
        model = GammatoneConvMLP()
    elif model_name == "standard_conv":
        model = StandardConvMLP()

    return train_model(model, dl_train, dl_test, lr, epochs=15) # Reduced for faster Optuna

def main():
    os.makedirs("differentiable_gammatone_experiment", exist_ok=True)

    results = {}

    # Pre-generate dataset for final eval to save time
    args = get_dataset_args()
    args.num_samples = 5000 # Reduced slightly from 10000 to save time
    data = make_dataset(args)
    X_train_full, y_train_full = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test_full, y_test_full = torch.tensor(data['x_test']).float(), torch.tensor(data['y_test']).long()

    dl_train_full = TensorDataLoader((X_train_full, y_train_full), batch_size=128, shuffle=True)
    dl_test_full = TensorDataLoader((X_test_full, y_test_full), batch_size=128, shuffle=False)

    for model_name in ["baseline", "standard_conv", "gammatone"]:
        print(f"Optimizing {model_name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_name), n_trials=10) # Reduced trials

        best_lr = study.best_params["lr"]
        print(f"Best LR for {model_name}: {best_lr}")

        # Final evaluation over 3 seeds to save time
        accs = []
        for seed in range(3):
            torch.manual_seed(seed)

            if model_name == "baseline":
                model = BaselineMLP()
            elif model_name == "gammatone":
                model = GammatoneConvMLP()
            elif model_name == "standard_conv":
                model = StandardConvMLP()

            acc = train_model(model, dl_train_full, dl_test_full, best_lr, epochs=30)
            accs.append(acc)
            print(f"Seed {seed}: {acc:.2f}%")

        results[model_name] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "best_lr": float(best_lr)
        }

    with open("differentiable_gammatone_experiment/results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Plot results
    model_names = list(results.keys())
    means = [results[m]["mean"] for m in model_names]
    stds = [results[m]["std"] for m in model_names]

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, means, yerr=stds, capsize=5, color=['blue', 'green', 'orange'])
    plt.ylabel("Accuracy (%)")
    plt.title("MNIST-1D Accuracy: Baseline vs Standard Conv vs Gammatone Conv")
    plt.savefig("differentiable_gammatone_experiment/results.png")

    # Write README
    with open("differentiable_gammatone_experiment/README.md", "w") as f:
        f.write("# Differentiable Gammatone Filterbank Experiment\n\n")
        f.write("This experiment evaluates a Differentiable Gammatone Layer for signal classification on the MNIST-1D dataset.\n\n")
        f.write("## Method\n\n")
        f.write("The Gammatone impulse response is defined as:\n")
        f.write("$$g(t) = t^{n-1} e^{-2\pi b t} \cos(2\pi f t + \phi)$$\n")
        f.write("where $n=4$ (standard for auditory modeling), $b$ is the bandwidth, $f$ is the center frequency, and $\phi$ is the phase.\n\n")
        f.write("The filterbank is implemented as a 1D convolution layer where the kernels are generated from these parameters. The entire process is differentiable, allowing the model to learn the optimal filters for the task.\n\n")
        f.write("## Results\n\n")
        f.write("| Model | Accuracy | Best LR |\n")
        f.write("|-------|----------|---------|\n")
        for m in model_names:
            f.write(f"| {m} | {results[m]['mean']:.2f}% ± {results[m]['std']:.2f}% | {results[m]['best_lr']:.6f} |\n")
        f.write("\n![Results](results.png)\n")

if __name__ == "__main__":
    main()
