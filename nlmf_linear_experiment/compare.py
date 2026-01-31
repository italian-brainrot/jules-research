import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import optuna
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
import os
import time

from layers import NLMFLinear, LowRankLinear, KroneckerLinear, DenseLinear

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test'])
    return X_train, y_train, X_test, y_test

class SimpleMLP(nn.Module):
    def __init__(self, layer_type, in_dim=40, hidden_dim=100, out_dim=10, rank=4):
        super().__init__()
        if layer_type == 'dense':
            self.l1 = DenseLinear(in_dim, hidden_dim)
            self.l2 = DenseLinear(hidden_dim, out_dim)
        elif layer_type == 'lowrank':
            self.l1 = LowRankLinear(in_dim, hidden_dim, rank=rank)
            self.l2 = LowRankLinear(hidden_dim, out_dim, rank=rank)
        elif layer_type == 'nlmf':
            self.l1 = NLMFLinear(in_dim, hidden_dim, rank=rank, hidden_dim=16)
            self.l2 = NLMFLinear(hidden_dim, out_dim, rank=rank, hidden_dim=16)
        elif layer_type == 'kronecker':
            # in=40, out=100 -> 4x10, 10x10
            self.l1 = KroneckerLinear(10, 4, 10, 10)
            # in=100, out=10 -> 2x10, 5x10
            self.l2 = KroneckerLinear(2, 10, 5, 10)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.l2(x)
        return x

def train_and_eval(model, dl_train, dl_test, lr, epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in dl_test:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        history.append({
            'train_loss': train_loss / len(dl_train),
            'test_loss': test_loss / len(dl_test),
            'test_acc': acc
        })
    return history

def objective(trial, layer_type, X_train, y_train, X_test, y_test, rank):
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    model = SimpleMLP(layer_type, rank=rank)
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    history = train_and_eval(model, dl_train, dl_test, lr, epochs=20)
    return history[-1]['test_acc']

def main():
    X_train, y_train, X_test, y_test = get_data()

    results = {}
    layer_types = [
        ('dense', 0),
        ('lowrank', 8),
        ('nlmf', 4),
        ('kronecker', 0)
    ]

    for layer_type, rank in layer_types:
        print(f"--- Tuning {layer_type} (rank={rank}) ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, layer_type, X_train, y_train, X_test, y_test, rank), n_trials=15)

        best_lr = study.best_params['lr']
        print(f"Best LR for {layer_type}: {best_lr}")

        print(f"--- Final Training {layer_type} ---")
        model = SimpleMLP(layer_type, rank=rank)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameter count: {param_count}")

        dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
        dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

        history = train_and_eval(model, dl_train, dl_test, best_lr, epochs=50)
        results[layer_type] = {
            'history': history,
            'params': param_count,
            'best_lr': best_lr,
            'final_acc': history[-1]['test_acc']
        }

    # Plot results
    plt.figure(figsize=(12, 8))
    for layer_type, data in results.items():
        accs = [h['test_acc'] for h in data['history']]
        plt.plot(accs, label=f"{layer_type} ({data['params']} params, acc={data['final_acc']:.2f}%)")
    plt.title("MNIST-1D Classification with Parameter-Efficient Layers")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("nlmf_linear_experiment/comparison_results.png")

    # Save summary
    with open("nlmf_linear_experiment/summary.txt", "w") as f:
        for layer_type, data in results.items():
            f.write(f"Model: {layer_type}\n")
            f.write(f"Parameters: {data['params']}\n")
            f.write(f"Best LR: {data['best_lr']}\n")
            f.write(f"Final Accuracy: {data['final_acc']:.2f}%\n")
            f.write("-" * 20 + "\n")

if __name__ == "__main__":
    main()
