import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mnist1d
import optuna
import numpy as np
import os
from model import BaselineMLP, ScaleTransformMLP, ScaleTransformAugmentedMLP, FourierMellinMLP

def get_data():
    args = mnist1d.data.get_dataset_args()
    data = mnist1d.data.get_dataset(args)

    x_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    x_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    return train_ds, test_ds

def train_model(model, train_loader, lr, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = model(batch_x)
            _, predicted = torch.max(output.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    return correct / total

def objective(trial, model_class, input_size, hidden_size, output_size, train_loader, test_loader):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    model = model_class(input_size, hidden_size, output_size)
    train_model(model, train_loader, lr, epochs=20)
    accuracy = evaluate_model(model, test_loader)
    return accuracy

def run_experiment():
    input_size = 40
    hidden_size = 256
    output_size = 10
    batch_size = 128

    train_ds, test_ds = get_data()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    models_to_test = {
        "BaselineMLP": BaselineMLP,
        "ScaleTransformMLP": ScaleTransformMLP,
        "ScaleTransformAugmentedMLP": ScaleTransformAugmentedMLP,
        "FourierMellinMLP": FourierMellinMLP
    }

    results = {}

    for name, model_class in models_to_test.items():
        print(f"Tuning {name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_class, input_size, hidden_size, output_size, train_loader, test_loader), n_trials=10)

        best_lr = study.best_params["lr"]
        print(f"Best LR for {name}: {best_lr}")

        # Evaluate with 3 seeds
        seeds = [42, 43, 44]
        accuracies = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = model_class(input_size, hidden_size, output_size)
            train_model(model, train_loader, best_lr, epochs=50)
            acc = evaluate_model(model, test_loader)
            accuracies.append(acc)
            print(f"Seed {seed}, Accuracy: {acc:.4f}")

        results[name] = {
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
            "best_lr": best_lr
        }

    # Save results
    with open("differentiable_scale_transform_experiment/results.txt", "w") as f:
        for name, res in results.items():
            f.write(f"{name}: Mean Accuracy = {res['mean']:.4f} +/- {res['std']:.4f}, Best LR = {res['best_lr']:.6f}\n")

    print("\nFinal Results:")
    for name, res in results.items():
        print(f"{name}: {res['mean']:.4f} +/- {res['std']:.4f}")

if __name__ == "__main__":
    run_experiment()
