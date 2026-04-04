import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import numpy as np
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import BaselineMLP, DVGMLP, DVGGNN

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test'])
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=10, batch_size=256):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
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

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    if model_type == "Baseline":
        model = BaselineMLP()
    elif model_type == "DVGMLP":
        initial_scale = trial.suggest_float("initial_scale", 1.0, 50.0)
        model = DVGMLP(initial_scale=initial_scale)
    elif model_type == "DVGGNN":
        initial_scale = trial.suggest_float("initial_scale", 1.0, 50.0)
        model = DVGGNN(initial_scale=initial_scale)
    else:
        raise ValueError("Unknown model type")

    # Using a subset for faster tuning
    return train_model(model, X_train[:2000], y_train[:2000], X_test[:500], y_test[:500], lr, epochs=5)

def main():
    X_train_full, y_train_full, X_test_full, y_test_full = get_data()

    # Use subset for everything to ensure it finishes
    X_train = X_train_full[:4000]
    y_train = y_train_full[:4000]
    X_test = X_test_full[:1000]
    y_test = y_test_full[:1000]

    results = {}

    # Reduced trials and epochs for faster execution in this environment
    n_trials = 2
    epochs_final = 10

    for model_type in ["Baseline", "DVGMLP", "DVGGNN"]:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=n_trials)

        best_lr = study.best_params["lr"]
        print(f"Best LR for {model_type}: {best_lr}")

        # Train 2 times with best params and average
        accs = []
        for i in range(2):
            if model_type == "Baseline":
                model = BaselineMLP()
            elif model_type == "DVGMLP":
                model = DVGMLP(initial_scale=study.best_params["initial_scale"])
            elif model_type == "DVGGNN":
                model = DVGGNN(initial_scale=study.best_params["initial_scale"])

            acc = train_model(model, X_train, y_train, X_test, y_test, best_lr, epochs=epochs_final)
            accs.append(acc)

        results[model_type] = {
            "mean": np.mean(accs),
            "std": np.std(accs),
            "best_params": study.best_params
        }
        print(f"Accuracy for {model_type}: {results[model_type]['mean']:.4f} ± {results[model_type]['std']:.4f}")

    import os
    output_dir = "differentiable_visibility_graph_experiment"
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        for model_type, res in results.items():
            f.write(f"{model_type}: {res['mean']:.4f} ± {res['std']:.4f}\n")
            f.write(f"Best Params: {res['best_params']}\n\n")

    # Plot results
    model_names = list(results.keys())
    means = [results[m]["mean"] for m in model_names]
    stds = [results[m]["std"] for m in model_names]

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, means, yerr=stds, capsize=5)
    plt.ylabel("Accuracy")
    plt.title("MNIST-1D Classification: DVG vs Baseline")
    plt.savefig(os.path.join(output_dir, "comparison.png"))

    # Visualize a learned DVG
    model = DVGMLP(initial_scale=results["DVGMLP"]["best_params"]["initial_scale"])
    train_model(model, X_train, y_train, X_test, y_test, results["DVGMLP"]["best_params"]["lr"], epochs=10)

    with torch.no_grad():
        A = model.dvg(X_test[:1])

    plt.figure(figsize=(8, 8))
    plt.imshow(A[0].cpu().numpy(), cmap='viridis')
    plt.title("Learned Soft Visibility Adjacency Matrix")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "dvg_viz.png"))

if __name__ == "__main__":
    main()
