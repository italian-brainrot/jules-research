import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import BaselineMLP, DFANet, DFAAugmentedMLP
import matplotlib.pyplot as plt

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
    elif model_type == "dfa":
        model = DFANet(input_dim=40, hidden_dim=hidden_dim, output_dim=10)
    elif model_type == "dfa_augmented":
        model = DFAAugmentedMLP(input_dim=40, hidden_dim=hidden_dim, output_dim=10)

    return train_model(model, X_train, y_train, X_test, y_test, lr)

def main():
    X_train, y_train, X_test, y_test = get_data()

    results = {}

    for model_type in ["baseline", "dfa", "dfa_augmented"]:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=10)

        best_acc = study.best_value
        results[model_type] = {
            "best_acc": best_acc,
            "best_lr": study.best_params["lr"]
        }
        print(f"Best accuracy for {model_type}: {best_acc:.4f} with LR: {study.best_params['lr']:.6f}")

    # Final evaluation with multiple seeds for the best models
    final_results = {}
    num_seeds = 3
    for model_type, res in results.items():
        accs = []
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            hidden_dim = 256
            if model_type == "baseline":
                model = BaselineMLP(input_dim=40, hidden_dim=hidden_dim, output_dim=10)
            elif model_type == "dfa":
                model = DFANet(input_dim=40, hidden_dim=hidden_dim, output_dim=10)
            elif model_type == "dfa_augmented":
                model = DFAAugmentedMLP(input_dim=40, hidden_dim=hidden_dim, output_dim=10)

            acc = train_model(model, X_train, y_train, X_test, y_test, res["best_lr"])
            accs.append(acc)
        final_results[model_type] = (np.mean(accs), np.std(accs))

    with open("differentiable_dfa_experiment/results.txt", "w") as f:
        for model_type, (mean, std) in final_results.items():
            f.write(f"{model_type}: {mean:.4f} +/- {std:.4f}\n")

    # Plotting
    labels = list(final_results.keys())
    means = [final_results[l][0] for l in labels]
    stds = [final_results[l][1] for l in labels]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, means, yerr=stds, capsize=5, color=['blue', 'green', 'orange'])
    plt.ylabel('Accuracy')
    plt.title('DFA Experiment Results on MNIST-1D')
    plt.savefig('differentiable_dfa_experiment/results.png')

if __name__ == "__main__":
    main()
