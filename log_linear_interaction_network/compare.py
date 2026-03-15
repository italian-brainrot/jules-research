import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from log_linear_interaction_network.model import BaselineMLP, LLIN
import numpy as np
import matplotlib.pyplot as plt
import ast
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.int64)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.int64)

    return X_train, y_train, X_test, y_test

def train_and_eval(model_class, input_dim, output_dim, params, X_train, y_train, X_test, y_test, epochs=50, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_class(input_dim, output_dim, params['hidden_dim'], params['n_layers'])
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    train_loader = TensorDataLoader((X_train, y_train), batch_size=params['batch_size'], shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=params['batch_size'], shuffle=False)

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        test_acc = correct / total
        history.append(test_acc)
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Test Acc: {test_acc:.4f}")

    return history

def main():
    X_train, y_train, X_test, y_test = get_data()

    with open("log_linear_interaction_network/best_params_baseline.txt", "r") as f:
        baseline_params = ast.literal_eval(f.read())

    with open("log_linear_interaction_network/best_params_llin.txt", "r") as f:
        llin_params = ast.literal_eval(f.read())

    seeds = [42, 43, 44]
    epochs = 50

    baseline_histories = []
    llin_histories = []

    for seed in seeds:
        print(f"Training Baseline (Seed {seed})...")
        baseline_histories.append(train_and_eval(BaselineMLP, 40, 10, baseline_params, X_train, y_train, X_test, y_test, epochs, seed))

        print(f"Training LLIN (Seed {seed})...")
        llin_histories.append(train_and_eval(LLIN, 40, 10, llin_params, X_train, y_train, X_test, y_test, epochs, seed))

    baseline_histories = np.array(baseline_histories)
    llin_histories = np.array(llin_histories)

    baseline_mean = baseline_histories.mean(axis=0)
    baseline_std = baseline_histories.std(axis=0)
    llin_mean = llin_histories.mean(axis=0)
    llin_std = llin_histories.std(axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(baseline_mean, label="Baseline MLP")
    plt.fill_between(range(epochs), baseline_mean - baseline_std, baseline_mean + baseline_std, alpha=0.2)
    plt.plot(llin_mean, label="LLIN (Log-Linear Interaction Network)")
    plt.fill_between(range(epochs), llin_mean - llin_std, llin_mean + llin_std, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Baseline MLP vs LLIN on MNIST-1D")
    plt.legend()
    plt.grid(True)
    plt.savefig("log_linear_interaction_network/comparison.png")

    final_baseline_acc = baseline_mean[-1]
    final_llin_acc = llin_mean[-1]

    print(f"\nFinal Baseline Accuracy: {final_baseline_acc:.4f} +/- {baseline_std[-1]:.4f}")
    print(f"Final LLIN Accuracy: {final_llin_acc:.4f} +/- {llin_std[-1]:.4f}")

    with open("log_linear_interaction_network/results.txt", "w") as f:
        f.write(f"Final Baseline Accuracy: {final_baseline_acc:.4f} +/- {baseline_std[-1]:.4f}\n")
        f.write(f"Final LLIN Accuracy: {final_llin_acc:.4f} +/- {llin_std[-1]:.4f}\n")

if __name__ == "__main__":
    main()
