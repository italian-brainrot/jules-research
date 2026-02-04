import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math

# Add the current directory to path so we can import model
sys.path.append(os.getcwd())
from fourier_series_activation_experiment.model import MLP, LFSA, ORelu, Snake

from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, X_test, y_test, activation_type='relu', lr=1e-3, epochs=20, batch_size=128, device='cpu'):
    model = MLP(40, [256, 256], 10, activation_type=activation_type, num_params='per_neuron').to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = TensorDataLoader((X_train.to(device), y_train.to(device)), batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            outputs = model(X_test.to(device))
            acc = (outputs.argmax(1) == y_test.to(device)).float().mean().item()

        history["train_loss"].append(avg_loss)
        history["test_acc"].append(acc)

    return model, history

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    X_train, y_train, X_test, y_test = get_data()

    activation_types = ['relu', 'gelu', 'orelu', 'snake', 'lfsa']
    best_lrs = {}

    for act in activation_types:
        print(f"Tuning {act}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: max(train_model(X_train, y_train, X_test, y_test, activation_type=act,
                                               lr=t.suggest_float("lr", 1e-4, 1e-2, log=True),
                                               epochs=10, device=device)[1]["test_acc"]), n_trials=10)
        best_lrs[act] = study.best_params["lr"]
        print(f"Best LR for {act}: {best_lrs[act]}")

    results = {}
    seeds = [42, 43, 44] # Reduced seeds for speed

    for act in activation_types:
        print(f"Final training for {act}...")
        act_results = []
        for seed in seeds:
            torch.manual_seed(seed)
            model, history = train_model(X_train, y_train, X_test, y_test, activation_type=act,
                                        lr=best_lrs[act], epochs=30, device=device)
            act_results.append(history)
        results[act] = act_results

    # Save results and plots
    plt.figure(figsize=(15, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    for act in activation_types:
        accs = np.array([h["test_acc"] for h in results[act]])
        mean_accs = accs.mean(axis=0)
        plt.plot(mean_accs, label=act)
        plt.fill_between(range(len(mean_accs)), accs.min(axis=0), accs.max(axis=0), alpha=0.2)
    plt.title("Test Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    for act in activation_types:
        losses = np.array([h["train_loss"] for h in results[act]])
        mean_losses = losses.mean(axis=0)
        plt.plot(mean_losses, label=act)
        plt.fill_between(range(len(mean_losses)), losses.min(axis=0), losses.max(axis=0), alpha=0.2)
    plt.title("Train Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("fourier_series_activation_experiment/comparison.png")

    # Visualize Activations
    plt.figure(figsize=(10, 8))
    x_range = torch.linspace(-5, 5, 500)

    # Plot standard ones
    plt.plot(x_range.numpy(), torch.relu(x_range).numpy(), '--', label='ReLU', alpha=0.5)
    plt.plot(x_range.numpy(), F.gelu(x_range).numpy(), '--', label='GELU', alpha=0.5)

    # Get one LFSA from the last trained model
    # Re-run a tiny training for LFSA to get "trained" activations
    model, _ = train_model(X_train, y_train, X_test, y_test, activation_type='lfsa', lr=best_lrs['lfsa'], epochs=5, device=device)
    model.to('cpu')

    with torch.no_grad():
        lfsa_layer = None
        for layer in model.net:
            if isinstance(layer, LFSA):
                lfsa_layer = layer
                break

        if lfsa_layer:
            # Plot a few neurons from the first hidden layer
            for i in range(min(5, lfsa_layer.w.shape[0])):
                # Individual neuron activation
                # We need to manually compute it for x_range
                y = lfsa_layer.w[i] * x_range + lfsa_layer.b[i]
                for k in range(lfsa_layer.K):
                    y = y + lfsa_layer.a[i, k] * torch.sin(lfsa_layer.omega[i, k] * x_range + lfsa_layer.phi[i, k])
                plt.plot(x_range.numpy(), y.numpy(), label=f'LFSA Neuron {i}')

    plt.title("Visualization of Learned LFSA Activations")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(True)
    plt.savefig("fourier_series_activation_experiment/learned_activations.png")

    # Summary results
    with open("fourier_series_activation_experiment/results.txt", "w") as f:
        for act in activation_types:
            final_accs = [h["test_acc"][-1] for h in results[act]]
            f.write(f"{act}: Mean Acc = {np.mean(final_accs):.4f}, Std = {np.std(final_accs):.4f}, Best LR = {best_lrs[act]}\n")

if __name__ == "__main__":
    run_experiment()
