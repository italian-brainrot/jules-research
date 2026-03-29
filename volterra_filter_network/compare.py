import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import VolterraMLP, BaselineMLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test'])
    return X_train, y_train, X_test, y_test

def train_and_evaluate(model_type, lr, X_train, y_train, X_test, y_test, seed, epochs=40):
    torch.manual_seed(seed)
    if model_type == "volterra":
        model = VolterraMLP(hidden_dim=40)
    else:
        model = BaselineMLP(hidden_dim=256)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)

    accuracies = []
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
            acc = (torch.argmax(outputs, dim=1) == y_test).float().mean().item()
            accuracies.append(acc)

    return accuracies

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    # Read best LRs
    lrs = {}
    with open("volterra_filter_network/best_params.txt", "r") as f:
        for line in f:
            k, v = line.strip().split(": ")
            lrs[k] = float(v)

    seeds = [42, 43, 44, 45, 46]
    volterra_results = []
    baseline_results = []

    print("Running VolterraMLP experiments...")
    for seed in seeds:
        accs = train_and_evaluate("volterra", lrs["volterra_lr"], X_train, y_train, X_test, y_test, seed)
        volterra_results.append(accs)
        print(f"Seed {seed} final accuracy: {accs[-1]:.4f}")

    print("\nRunning BaselineMLP experiments...")
    for seed in seeds:
        accs = train_and_evaluate("baseline", lrs["baseline_lr"], X_train, y_train, X_test, y_test, seed)
        baseline_results.append(accs)
        print(f"Seed {seed} final accuracy: {accs[-1]:.4f}")

    volterra_results = np.array(volterra_results)
    baseline_results = np.array(baseline_results)

    v_mean = np.mean(volterra_results, axis=0)
    v_std = np.std(volterra_results, axis=0)
    b_mean = np.mean(baseline_results, axis=0)
    b_std = np.std(baseline_results, axis=0)

    plt.figure(figsize=(10, 6))
    epochs = range(1, 41)
    plt.plot(epochs, v_mean, label=f"VolterraMLP (Final: {v_mean[-1]:.4f} ± {v_std[-1]:.4f})")
    plt.fill_between(epochs, v_mean - v_std, v_mean + v_std, alpha=0.2)
    plt.plot(epochs, b_mean, label=f"BaselineMLP (Final: {b_mean[-1]:.4f} ± {b_std[-1]:.4f})")
    plt.fill_between(epochs, b_mean - b_std, b_mean + b_std, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("VolterraMLP vs BaselineMLP on MNIST1D")
    plt.legend()
    plt.grid(True)
    plt.savefig("volterra_filter_network/results.png")

    with open("volterra_filter_network/results.txt", "w") as f:
        f.write(f"VolterraMLP Final Accuracy: {v_mean[-1]:.4f} ± {v_std[-1]:.4f}\n")
        f.write(f"BaselineMLP Final Accuracy: {b_mean[-1]:.4f} ± {b_std[-1]:.4f}\n")
