import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import optuna
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from optimizer import FIWDAdamW
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data(label_noise=0.2):
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)

    x_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    x_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    if label_noise > 0:
        n_noisy = int(label_noise * len(y_train))
        noisy_idx = torch.randperm(len(y_train))[:n_noisy]
        y_train[noisy_idx] = torch.randint(0, 10, (n_noisy,))

    # Split train into train and val
    n_train = int(0.8 * len(x_train))
    x_val = x_train[n_train:]
    y_val = y_train[n_train:]
    x_train = x_train[:n_train]
    y_train = y_train[:n_train]

    return x_train, y_train, x_val, y_val, x_test, y_test

class MLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
    return correct / loader.data_length()

def objective(trial, mode, data):
    x_train, y_train, x_val, y_val = data
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-5, 1.0, log=True)

    model = MLP().to(device)

    if mode == 'FIWD':
        gamma = trial.suggest_float('gamma', 0.1, 2.0)
        tau = trial.suggest_float('tau', 1e-6, 1e-2, log=True)
        optimizer = FIWDAdamW(model.parameters(), lr=lr, weight_decay=wd, gamma=gamma, tau=tau)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_val_acc = 0
    for epoch in range(20):
        train_epoch(model, train_loader, optimizer)
        val_acc = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def main():
    label_noise = 0.2
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(label_noise=label_noise)
    data_for_obj = (x_train, y_train, x_val, y_val)

    modes = ['AdamW', 'FIWD']
    best_params = {}

    for mode in modes:
        print(f"--- Optimizing {mode} ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode, data_for_obj), n_trials=30)
        best_params[mode] = study.best_params
        print(f"Best params for {mode}: {study.best_params}")

    # Final comparison
    results = {mode: [] for mode in modes}
    histories = {mode: [] for mode in modes}

    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)
    test_loader = TensorDataLoader((x_test, y_test), batch_size=128, shuffle=False)

    seeds = [42, 43, 44]
    epochs = 50

    for mode in modes:
        params = best_params[mode]
        for seed in seeds:
            set_seed(seed)
            model = MLP().to(device)
            if mode == 'FIWD':
                optimizer = FIWDAdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'],
                                      gamma=params['gamma'], tau=params['tau'])
            else:
                optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

            history = []
            best_val_acc = 0
            test_acc_at_best_val = 0

            for epoch in range(epochs):
                train_loss = train_epoch(model, train_loader, optimizer)
                val_acc = evaluate(model, val_loader)
                test_acc = evaluate(model, test_loader)
                history.append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc_at_best_val = test_acc

            results[mode].append(test_acc_at_best_val)
            histories[mode].append(history)
            print(f"Mode {mode}, Seed {seed}, Best Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc_at_best_val:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    for mode in modes:
        avg_history = np.mean(histories[mode], axis=0)
        plt.plot(avg_history, label=mode)
    plt.title(f'Validation Accuracy (Label Noise: {label_noise})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('fisher_inverse_weight_decay_experiment/results.png')

    # Save summary
    with open('fisher_inverse_weight_decay_experiment/results.txt', 'w') as f:
        f.write(f"Label Noise: {label_noise}\n")
        for mode in modes:
            mean_acc = np.mean(results[mode])
            std_acc = np.std(results[mode])
            f.write(f"{mode}: {mean_acc:.4f} +/- {std_acc:.4f}\n")
            f.write(f"Best Params: {best_params[mode]}\n")

if __name__ == '__main__':
    main()
