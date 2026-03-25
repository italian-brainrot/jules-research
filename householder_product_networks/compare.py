import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from householder_product_networks.models import get_model, count_parameters
import os
import ast

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_and_eval(model_name, lr, X_train, y_train, X_test, y_test, epochs=50, seed=42):
    torch.manual_seed(seed)
    input_size = 40
    hidden_size = 64
    output_size = 10

    if model_name == "householder":
        model = get_model(model_name, input_size, hidden_size, output_size, num_layers=2, num_reflectors=32)
    else:
        model = get_model(model_name, input_size, hidden_size, output_size, num_layers=2)

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in dl_test:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        acc = 100 * correct / total
        history.append(acc)
        print(f"Seed {seed}, Model {model_name}, Epoch {epoch+1}/{epochs}, Acc: {acc:.2f}%")

    return history, count_parameters(model)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    with open("householder_product_networks/best_params_baseline.txt", "r") as f:
        baseline_params = ast.literal_eval(f.read())
    with open("householder_product_networks/best_params_householder.txt", "r") as f:
        householder_params = ast.literal_eval(f.read())

    seeds = [42, 43, 44]
    results = {}
    params_count = {}

    for model_name, lr in [("baseline", baseline_params['lr']), ("householder", householder_params['lr'])]:
        all_histories = []
        for seed in seeds:
            history, p_count = train_and_eval(model_name, lr, X_train, y_train, X_test, y_test, seed=seed)
            all_histories.append(history)
            params_count[model_name] = p_count
        results[model_name] = np.array(all_histories)

    # Plot results
    plt.figure(figsize=(10, 6))
    for model_name, histories in results.items():
        mean_hist = np.mean(histories, axis=0)
        std_hist = np.std(histories, axis=0)
        epochs = np.arange(1, len(mean_hist) + 1)
        plt.plot(epochs, mean_hist, label=f"{model_name} ({params_count[model_name]} params)")
        plt.fill_between(epochs, mean_hist - std_hist, mean_hist + std_hist, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Householder vs Baseline MLP on MNIST-1D")
    plt.legend()
    plt.grid(True)
    plt.savefig("householder_product_networks/comparison_results.png")

    # Save summary
    with open("householder_product_networks/summary.txt", "w") as f:
        for model_name, histories in results.items():
            final_accs = histories[:, -1]
            f.write(f"{model_name}: {np.mean(final_accs):.2f}% +/- {np.std(final_accs):.2f}%\n")
            f.write(f"Parameters: {params_count[model_name]}\n")
