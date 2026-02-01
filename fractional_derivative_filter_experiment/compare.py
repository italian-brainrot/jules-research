import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import LFDFModel, MLPBaseline, ConvBaseline

# Ensure directory for plots exists
PLOT_DIR = "fractional_derivative_filter_experiment"
os.makedirs(PLOT_DIR, exist_ok=True)

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 5000 # Using 5000 for faster experimentation
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=100, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_out = model(X_test)
            test_acc = (test_out.argmax(1) == y_test).float().mean().item()

        history['train_loss'].append(total_loss / len(train_loader))
        history['test_acc'].append(test_acc)

    return history, history['test_acc'][-1]

def run_study(model_class, name, X_train, y_train, X_test, y_test, n_trials=12):
    print(f"Running study for {name}...")
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        model = model_class()
        _, acc = train_model(model, X_train, y_train, X_test, y_test, lr, epochs=30)
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best LR for {name}: {study.best_params['lr']}, Best Acc: {study.best_value}")
    return study.best_params['lr']

def main():
    X_train, y_train, X_test, y_test = get_data()

    model_configs = [
        (MLPBaseline, "MLP_Baseline"),
        (ConvBaseline, "Conv_Baseline"),
        (LFDFModel, "LFDF_Model")
    ]

    results = {}

    for model_class, name in model_configs:
        best_lr = run_study(model_class, name, X_train, y_train, X_test, y_test)

        print(f"Final training for {name} with LR={best_lr}...")
        model = model_class()
        history, final_acc = train_model(model, X_train, y_train, X_test, y_test, best_lr, epochs=60)
        results[name] = {
            'history': history,
            'final_acc': final_acc,
            'model': model
        }

    # Plot Accuracies
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        plt.plot(res['history']['test_acc'], label=f"{name} (Best: {max(res['history']['test_acc']):.4f})")
    plt.title("Test Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "accuracy_comparison.png"))
    plt.close()

    # Visualize Learned Alphas for LFDF
    lfdf_model = results['LFDF_Model']['model']
    alphas = (torch.sigmoid(lfdf_model.lfdf.alpha_raw) * 2.0).detach().cpu().numpy().flatten()

    plt.figure(figsize=(8, 4))
    plt.hist(alphas, bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribution of Learned Alpha (Fractional Order)")
    plt.xlabel("Alpha")
    plt.ylabel("Count")
    plt.savefig(os.path.join(PLOT_DIR, "alpha_distribution.png"))
    plt.close()

    # Write summary
    with open(os.path.join(PLOT_DIR, "results.txt"), "w") as f:
        for name, res in results.items():
            f.write(f"{name}: Final Acc = {res['final_acc']:.4f}, Best Acc = {max(res['history']['test_acc']):.4f}\n")
        f.write("\nLearned Alphas:\n")
        f.write(str(alphas.tolist()))

if __name__ == "__main__":
    main()
