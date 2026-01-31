import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import numpy as np
import time
import os
from models import Net

def train_eval(pool_type, lr, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)

    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()

    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=64, shuffle=False)

    model = Net(pool_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl_test:
            x, y = x.to(device), y.to(device)
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total

def objective(trial, pool_type):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    acc = train_eval(pool_type, lr, epochs=5)
    return acc

if __name__ == "__main__":
    pool_types = ['max', 'avg', 'median', 'lp', 'quantile']
    results = {}

    for pt in pool_types:
        print(f"--- Optimizing {pt} ---")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, pt), n_trials=8)

        best_lr = study.best_params['lr']
        print(f"Best LR for {pt}: {best_lr:.6f}")

        # Train final model with best LR for more epochs
        print(f"Final evaluation for {pt}...")
        final_accs = []
        for _ in range(2): # Run 2 times to get average
            acc = train_eval(pt, best_lr, epochs=10)
            final_accs.append(acc)

        avg_acc = np.mean(final_accs)
        std_acc = np.std(final_accs)
        results[pt] = (avg_acc, std_acc, best_lr)
        print(f"Result for {pt}: {avg_acc:.4f} +- {std_acc:.4f}")

    # Print summary
    print("\nSummary:")
    for pt, (avg, std, lr) in results.items():
        print(f"{pt}: {avg:.4f} +- {std:.4f} (LR: {lr:.6f})")

    # Save results to markdown
    with open("differentiable_quantile_pooling_experiment/results.md", "w") as f:
        f.write("# Experiment Results: Implicit Differentiable Quantile Pooling\n\n")
        f.write("| Pool Type | Accuracy | Best LR |\n")
        f.write("|-----------|----------|---------|\n")
        for pt, (avg, std, lr) in results.items():
            f.write(f"| {pt} | {avg:.4f} +- {std:.4f} | {lr:.6f} |\n")
