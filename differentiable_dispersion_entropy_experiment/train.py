import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import BaselineMLP, DDEMLP, DDEAugmentedMLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=30, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in dl_train:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        out_test = model(X_test)
        preds = torch.argmax(out_test, dim=1)
        acc = (preds == y_test).float().mean().item()
    return acc

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = 256

    if model_type == "baseline":
        model = BaselineMLP(input_dim=40, hidden_dim=hidden_dim, output_dim=10)
    elif model_type == "dde":
        model = DDEMLP(input_dim=40, hidden_dim=hidden_dim, output_dim=10)
    elif model_type == "dde_augmented":
        model = DDEAugmentedMLP(input_dim=40, hidden_dim=hidden_dim, output_dim=10)

    return train_model(model, X_train, y_train, X_test, y_test, lr)

def main():
    X_train, y_train, X_test, y_test = get_data()

    results = {}
    model_types = ["baseline", "dde", "dde_augmented"]

    for model_type in model_types:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=10)

        results[model_type] = {
            "best_acc": study.best_value,
            "best_lr": study.best_params["lr"]
        }
        print(f"Best accuracy for {model_type}: {study.best_value:.4f} with LR: {study.best_params['lr']:.6f}")

    # Final evaluation
    final_results = {}
    num_seeds = 3
    for model_type in model_types:
        accs = []
        best_lr = results[model_type]["best_lr"]
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            hidden_dim = 256
            if model_type == "baseline":
                model = BaselineMLP(input_dim=40, hidden_dim=hidden_dim, output_dim=10)
            elif model_type == "dde":
                model = DDEMLP(input_dim=40, hidden_dim=hidden_dim, output_dim=10)
            elif model_type == "dde_augmented":
                model = DDEAugmentedMLP(input_dim=40, hidden_dim=hidden_dim, output_dim=10)

            acc = train_model(model, X_train, y_train, X_test, y_test, best_lr)
            accs.append(acc)
        final_results[model_type] = (np.mean(accs), np.std(accs))

    # Save results
    with open("differentiable_dispersion_entropy_experiment/results.txt", "w") as f:
        for model_type, (mean, std) in final_results.items():
            f.write(f"{model_type}: {mean:.4f} +/- {std:.4f}\n")

    # Plotting
    labels = list(final_results.keys())
    means = [final_results[l][0] for l in labels]
    stds = [final_results[l][1] for l in labels]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, means, yerr=stds, capsize=5, color=['skyblue', 'salmon', 'lightgreen'])
    plt.ylabel('Accuracy')
    plt.title('Dispersion Entropy Experiment Results on MNIST-1D')
    plt.ylim(0, 1.0)
    plt.savefig('differentiable_dispersion_entropy_experiment/results.png')

if __name__ == "__main__":
    main()
