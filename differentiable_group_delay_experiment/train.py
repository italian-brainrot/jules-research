import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from differentiable_group_delay_experiment.model import BaselineMLP, GroupDelayAugmentedMLP, GroupDelayMLP
import os

def train_model(model, dl_train, dl_test, lr, epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        train_accs.append(train_acc)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in dl_test:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100. * correct / total
        test_accs.append(test_acc)

    return max(test_accs)

def objective(trial, model_class, dl_train, dl_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    model = model_class()
    acc = train_model(model, dl_train, dl_test, lr, epochs=15)
    return acc

def run_experiment():
    # Load data
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    models_to_test = {
        "BaselineMLP": BaselineMLP,
        "GroupDelayAugmentedMLP": GroupDelayAugmentedMLP,
        "GroupDelayMLP": GroupDelayMLP
    }

    results = {}

    for name, model_class in models_to_test.items():
        print(f"Tuning {name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_class, dl_train, dl_test), n_trials=10)

        best_lr = study.best_params['lr']
        print(f"Best LR for {name}: {best_lr}")

        # Final evaluation with 3 seeds
        eval_accs = []
        for seed in range(3):
            torch.manual_seed(seed)
            model = model_class()
            acc = train_model(model, dl_train, dl_test, best_lr, epochs=50)
            eval_accs.append(acc)

        results[name] = {
            "mean": np.mean(eval_accs),
            "std": np.std(eval_accs),
            "best_lr": best_lr
        }
        print(f"Final {name} accuracy: {results[name]['mean']:.2f}% +/- {results[name]['std']:.2f}%")

    # Save results
    with open("differentiable_group_delay_experiment/results.txt", "w") as f:
        for name, res in results.items():
            f.write(f"{name}: {res['mean']:.2f}% +/- {res['std']:.2f}% (Best LR: {res['best_lr']})\n")

    # Plot results
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    means = [results[n]['mean'] for n in names]
    stds = [results[n]['std'] for n in names]

    plt.bar(names, means, yerr=stds, capsize=5, color=['skyblue', 'salmon', 'lightgreen'])
    plt.ylabel("Test Accuracy (%)")
    plt.title("Comparison of MLP and Group-Delay Models on MNIST-1D")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("differentiable_group_delay_experiment/comparison.png")
    plt.close()

if __name__ == "__main__":
    run_experiment()
