import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import AdaptiveMLP
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_eval(params, X_train, y_train, X_test, y_test, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptiveMLP(
        hidden_dim=256,
        p_base=params['p_base'],
        gamma=params['gamma']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    dl_train = TensorDataLoader((X_train.to(device), y_train.to(device)), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test.to(device), y_test.to(device)), batch_size=128, shuffle=False)

    test_accs = []

    for epoch in range(epochs):
        model.train()
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in dl_test:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        acc = correct / total
        test_accs.append(acc)

    return test_accs

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    with open('fisher_adaptive_dropout_experiment/best_params.json', 'r') as f:
        best_params = json.load(f)

    seeds = [42, 43, 44]

    modes = {
        'Baseline': {'gamma': 0.0, 'p_base': 0.2, 'lr': best_params['lr'], 'weight_decay': best_params['weight_decay']}, # A standard default
        'Fisher-Adaptive (Best)': best_params,
        'FTD (gamma=1.0)': {**best_params, 'gamma': 1.0},
        'FSD (gamma=-1.0)': {**best_params, 'gamma': -1.0}
    }

    results = {}

    for mode_name, params in modes.items():
        print(f"Running mode: {mode_name}")
        all_accs = []
        for seed in seeds:
            torch.manual_seed(seed)
            accs = train_eval(params, X_train, y_train, X_test, y_test)
            all_accs.append(accs)
        results[mode_name] = np.array(all_accs)

    plt.figure(figsize=(10, 6))
    for mode_name, accs in results.items():
        mean_accs = accs.mean(axis=0)
        std_accs = accs.std(axis=0)
        plt.plot(mean_accs, label=mode_name)
        plt.fill_between(range(len(mean_accs)), mean_accs - std_accs, mean_accs + std_accs, alpha=0.2)

    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Fisher-Adaptive Dropout Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('fisher_adaptive_dropout_experiment/comparison.png')

    # Save textual results
    with open('fisher_adaptive_dropout_experiment/results.txt', 'w') as f:
        for mode_name, accs in results.items():
            final_acc = accs[:, -1].mean()
            final_std = accs[:, -1].std()
            f.write(f"{mode_name}: {final_acc:.4f} +- {final_std:.4f}\n")
            f.write(f"Params: {modes[mode_name]}\n\n")
