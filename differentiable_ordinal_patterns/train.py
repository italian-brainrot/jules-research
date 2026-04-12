import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import DOPMLP, DOPAugmentedMLP, BaselineMLP
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

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=30, batch_size=64):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    return accuracy

def objective(trial, model_class, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = 256
    if model_class == BaselineMLP:
        model = model_class(input_dim=40, hidden_dim=hidden_dim, output_dim=10)
    else:
        model = model_class(input_dim=40, hidden_dim=hidden_dim, output_dim=10, d=3, tau=1)

    return train_model(model, X_train, y_train, X_test, y_test, lr, epochs=15)

def main():
    X_train, y_train, X_test, y_test = get_data()

    models_to_test = {
        "BaselineMLP": BaselineMLP,
        "DOPMLP": DOPMLP,
        "DOPAugmentedMLP": DOPAugmentedMLP
    }

    best_lrs = {}

    for name, model_class in models_to_test.items():
        print(f"Tuning {name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_class, X_train, y_train, X_test, y_test), n_trials=5)
        best_lrs[name] = study.best_params["lr"]
        print(f"Best LR for {name}: {best_lrs[name]}")

    results = {name: [] for name in models_to_test}
    seeds = [42, 43]

    for seed in seeds:
        torch.manual_seed(seed)
        for name, model_class in models_to_test.items():
            print(f"Evaluating {name} with seed {seed}...")
            if model_class == BaselineMLP:
                model = model_class(input_dim=40, hidden_dim=256, output_dim=10)
            else:
                model = model_class(input_dim=40, hidden_dim=256, output_dim=10, d=3, tau=1)

            acc = train_model(model, X_train, y_train, X_test, y_test, best_lrs[name], epochs=30)
            results[name].append(acc)
            print(f"Accuracy: {acc:.4f}")

    # Log results
    with open("differentiable_ordinal_patterns/results.txt", "w") as f:
        for name, accs in results.items():
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            f.write(f"{name}: {mean_acc:.4f} +/- {std_acc:.4f}\n")
            print(f"{name}: {mean_acc:.4f} +/- {std_acc:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    means = [np.mean(results[n]) for n in names]
    stds = [np.std(results[n]) for n in names]

    plt.bar(names, means, yerr=stds, capsize=5, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylabel('Accuracy')
    plt.title('DOP Model Comparison on MNIST-1D')
    plt.savefig('differentiable_ordinal_patterns/results.png')
    plt.close()

if __name__ == "__main__":
    main()
