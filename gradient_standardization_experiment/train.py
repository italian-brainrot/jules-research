import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import optuna
import json
import os
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data():
    args = get_dataset_args()
    args.num_samples = 4000
    data = make_dataset(args)

    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()

    # Split train into train and val
    n_train = int(0.8 * len(X_train))
    X_val = X_train[n_train:]
    y_val = y_train[n_train:]
    X_train_final = X_train[:n_train]
    y_train_final = y_train[:n_train]

    return X_train_final, y_train_final, X_val, y_val, X_test, y_test

class MLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def centralize_gradient(grad):
    if grad.dim() > 1:
        # For 2D (Linear layers), centralize along the output dimension (rows)
        # Weight shape: (out_features, in_features)
        mean = grad.mean(dim=1, keepdim=True)
        return grad - mean
    return grad

def standardize_gradient(grad):
    if grad.dim() > 1:
        # Centralize
        mean = grad.mean(dim=1, keepdim=True)
        grad = grad - mean
        # Scale (standardize variance across output neurons)
        std = grad.std(dim=1, keepdim=True) + 1e-8
        return grad / std
    return grad

def apply_gradient_modification(model, mode):
    if mode == 'Baseline':
        return
    for name, p in model.named_parameters():
        if p.grad is not None and 'weight' in name and p.dim() > 1:
            if mode == 'GC':
                p.grad.data = centralize_gradient(p.grad.data)
            elif mode == 'NGS':
                p.grad.data = standardize_gradient(p.grad.data)

def train_model(mode, config, data, epochs=50, seed=42):
    set_seed(seed)
    X_train, y_train, X_val, y_val, X_test, y_test = data
    train_loader = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    val_loader = TensorDataLoader((X_val, y_val), batch_size=64, shuffle=False)

    model = MLP().to('cpu')
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    history = {'train_loss': [], 'val_acc': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()

            # Apply GC or NGS
            apply_gradient_modification(model, mode)

            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                logits = model(inputs)
                preds = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()

        val_acc = val_correct / len(y_val)

        history['train_loss'].append(total_loss / len(train_loader))
        history['val_acc'].append(val_acc)

        # Test accuracy (tracked for analysis)
        test_correct = 0
        with torch.no_grad():
            logits = model(X_test)
            preds = logits.argmax(dim=1)
            test_correct = (preds == y_test).sum().item()
        test_acc = test_correct / len(y_test)
        history['test_acc'].append(test_acc)

    return max(history['val_acc']), history

def objective(trial, mode, data):
    config = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
    }
    val_acc, _ = train_model(mode, config, data, epochs=20, seed=42)
    return val_acc

def main():
    data = get_data()
    modes = ['Baseline', 'GC', 'NGS']
    best_configs = {}

    for mode in modes:
        print(f"Optimizing {mode}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode, data), n_trials=20)
        best_configs[mode] = study.best_params
        print(f"Best params for {mode}: {best_configs[mode]}")

    results = {}
    for mode in modes:
        print(f"Running final evaluation for {mode}...")
        mode_val_accs = []
        mode_test_accs = []
        mode_histories = []
        for seed in range(5):
            _, history = train_model(mode, best_configs[mode], data, epochs=100, seed=seed+100)
            mode_val_accs.append(max(history['val_acc']))
            mode_test_accs.append(history['test_acc'][np.argmax(history['val_acc'])])
            mode_histories.append(history)

        results[mode] = {
            'mean_val_acc': np.mean(mode_val_accs),
            'std_val_acc': np.std(mode_val_accs),
            'mean_test_acc': np.mean(mode_test_accs),
            'std_test_acc': np.std(mode_test_accs),
            'histories': mode_histories
        }

    # Save results
    serializable_results = {}
    for mode, data_res in results.items():
        serializable_results[mode] = {
            'mean_val_acc': float(data_res['mean_val_acc']),
            'std_val_acc': float(data_res['std_val_acc']),
            'mean_test_acc': float(data_res['mean_test_acc']),
            'std_test_acc': float(data_res['std_test_acc']),
            'best_params': best_configs[mode]
        }

    with open('gradient_standardization_experiment/results.json', 'w') as f:
        json.dump(serializable_results, f, indent=4)

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for mode in modes:
        all_losses = np.array([h['train_loss'] for h in results[mode]['histories']])
        mean_loss = all_losses.mean(axis=0)
        plt.plot(mean_loss, label=mode)
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for mode in modes:
        all_accs = np.array([h['val_acc'] for h in results[mode]['histories']])
        mean_accs = all_accs.mean(axis=0)
        plt.plot(mean_accs, label=mode)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('gradient_standardization_experiment/results.png')
    plt.close()

    print("\nFinal Results:")
    for mode in modes:
        print(f"{mode}: Val Acc = {results[mode]['mean_val_acc']:.4f} +/- {results[mode]['std_val_acc']:.4f}, Test Acc = {results[mode]['mean_test_acc']:.4f} +/- {results[mode]['std_test_acc']:.4f}")

if __name__ == "__main__":
    main()
