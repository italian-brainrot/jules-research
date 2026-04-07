import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import BaselineMLP, TropicalMLP, TropicalAugmentedMLP
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_and_eval(model, dl_train, dl_test, epochs=30, lr=1e-3, weight_decay=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for x, y in dl_train:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        train_accs.append(correct / total)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dl_test:
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        test_accs.append(correct / total)

    return train_accs, test_accs

def run_comparison():
    X_train, y_train, X_test, y_test = get_data()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    # Use best params found from Optuna (or close enough)
    models = {
        "Baseline": BaselineMLP(40, 128, 10, num_layers=4),
        "Tropical": TropicalMLP(40, 128, 10, num_layers=2, init_beta=1.0),
        "Augmented": TropicalAugmentedMLP(40, 128, 10, num_layers=2, init_beta=4.0)
    }

    lrs = {
        "Baseline": 2e-3,
        "Tropical": 5e-3,
        "Augmented": 2e-3
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        train_acc, test_acc = train_and_eval(model, dl_train, dl_test, lr=lrs[name])
        results[name] = test_acc
        print(f"{name} Best Test Acc: {max(test_acc):.4f}")

        # Log learned beta for Tropical layers
        if name == "Tropical":
            betas = [p.data.mean().item() for n, p in model.named_parameters() if "beta" in n]
            print(f"Mean learned betas: {betas}")
        elif name == "Augmented":
            beta = model.tropical.beta.data.mean().item()
            print(f"Mean learned beta: {beta}")

    plt.figure(figsize=(10, 6))
    for name, accs in results.items():
        plt.plot(accs, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Comparison of Tropical and Baseline Networks")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_plot.png")
    plt.close()

if __name__ == "__main__":
    run_comparison()
