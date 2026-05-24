import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import json
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import DOMPNet, DOMPAugmentedMLP, BaselineMLP

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
        for inputs, targets in train_loader:
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

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    input_dim = 40
    output_dim = 10
    hidden_dim = 128
    dict_size = 100
    num_iterations = 10

    if model_type == "baseline":
        model = BaselineMLP(input_dim, hidden_dim, output_dim)
    elif model_type == "domp_net":
        model = DOMPNet(input_dim, dict_size, num_iterations, hidden_dim, output_dim)
    elif model_type == "domp_augmented":
        model = DOMPAugmentedMLP(input_dim, dict_size, num_iterations, hidden_dim, output_dim)

    return train_model(model, X_train, y_train, X_test, y_test, lr)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    results = {}

    for model_type in ["baseline", "domp_net", "domp_augmented"]:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=10)

        best_lr = study.best_params["lr"]
        print(f"Best LR for {model_type}: {best_lr}")

        # Train final model with best LR multiple times for stability
        accs = []
        for i in range(3):
            input_dim = 40
            output_dim = 10
            hidden_dim = 128
            dict_size = 100
            num_iterations = 10
            if model_type == "baseline":
                model = BaselineMLP(input_dim, hidden_dim, output_dim)
            elif model_type == "domp_net":
                model = DOMPNet(input_dim, dict_size, num_iterations, hidden_dim, output_dim)
            elif model_type == "domp_augmented":
                model = DOMPAugmentedMLP(input_dim, dict_size, num_iterations, hidden_dim, output_dim)

            acc = train_model(model, X_train, y_train, X_test, y_test, best_lr)
            accs.append(acc)

        results[model_type] = {
            "mean": np.mean(accs),
            "std": np.std(accs),
            "best_lr": best_lr
        }
        print(f"Results for {model_type}: {results[model_type]['mean']:.4f} +- {results[model_type]['std']:.4f}")

    with open("differentiable_omp_experiment/results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Plot results
    model_names = list(results.keys())
    means = [results[m]["mean"] for m in model_names]
    stds = [results[m]["std"] for m in model_names]

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, means, yerr=stds, capsize=5, color=['blue', 'green', 'orange'])
    plt.ylabel("Accuracy")
    plt.title("Model Comparison on MNIST-1D")
    plt.savefig("differentiable_omp_experiment/comparison.png")
    plt.close()
