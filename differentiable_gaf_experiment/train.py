import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import GAFNet, CNN1D
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

def train_model(model, dl_train, lr, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y).sum().item() / y.size(0)
    return accuracy

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)

    if model_type == 'gaf':
        model = GAFNet()
    else:
        model = CNN1D()

    train_model(model, dl_train, lr, epochs=5) # Reduced epochs for tuning
    accuracy = evaluate_model(model, X_test, y_test)
    return accuracy

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)

    results = {}

    for model_type in ['cnn1d', 'gaf']:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=5) # Reduced trials
        best_lr = study.best_params['lr']
        print(f"Best LR for {model_type}: {best_lr}")

        # Final train with 2 seeds to save time
        seeds = [42, 43]
        accuracies = []
        for seed in seeds:
            torch.manual_seed(seed)
            if model_type == 'gaf':
                model = GAFNet()
            else:
                model = CNN1D()

            print(f"Training {model_type} with seed {seed}...")
            train_model(model, dl_train, best_lr, epochs=20) # Reduced epochs
            acc = evaluate_model(model, X_test, y_test)
            accuracies.append(acc)
            print(f"Accuracy: {acc}")

        results[model_type] = {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'best_lr': best_lr
        }

    # Write results
    with open("differentiable_gaf_experiment/results.txt", "w") as f:
        for model_type, res in results.items():
            f.write(f"{model_type}: Mean {res['mean']:.4f}, Std {res['std']:.4f}, Best LR {res['best_lr']:.6e}\n")

    # Plot results
    model_names = list(results.keys())
    means = [results[m]['mean'] for m in model_names]
    stds = [results[m]['std'] for m in model_names]

    plt.figure(figsize=(8, 6))
    plt.bar(model_names, means, yerr=stds, capsize=5, color=['blue', 'green'])
    plt.ylabel('Test Accuracy')
    plt.title('GAFNet vs 1D CNN Baseline on MNIST-1D')
    plt.ylim(0, 1.0)
    plt.savefig('differentiable_gaf_experiment/comparison.png')

    print("Experiment completed!")

if __name__ == "__main__":
    run_experiment()
