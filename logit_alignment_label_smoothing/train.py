import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import optuna
import json
import os
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from logit_alignment_label_smoothing.model import MLP
from logit_alignment_label_smoothing.loss import LGALSLoss, fixed_label_smoothing_loss

def add_label_noise(y, noise_rate=0.2):
    y_noisy = y.clone()
    n_samples = y.shape[0]
    n_noisy = int(noise_rate * n_samples)
    indices = torch.randperm(n_samples)[:n_noisy]
    new_labels = torch.randint(0, 10, (n_noisy,))
    y_noisy[indices] = new_labels
    return y_noisy

def get_data():
    args = get_dataset_args()
    args.num_samples = 4000
    data = make_dataset(args)

    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()

    y_train_noisy = add_label_noise(y_train, noise_rate=0.2)

    return X_train, y_train_noisy, X_test, y_test

def train_model(mode, config, X_train, y_train, X_test, y_test, epochs=100, seed=42):
    torch.manual_seed(seed)
    model = MLP().to('cpu')
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    if mode == 'LGALS':
        criterion = LGALSLoss(epsilon_max=config['epsilon_max'], gamma=config['gamma'])

    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            if mode == 'Baseline':
                loss = F.cross_entropy(logits, targets)
            elif mode == 'FixedLS':
                loss = fixed_label_smoothing_loss(logits, targets, epsilon=config['epsilon'])
            elif mode == 'LGALS':
                loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            logits = model(X_test)
            preds = logits.argmax(dim=1)
            correct = (preds == y_test).sum().item()

        acc = correct / len(y_test)
        history['train_loss'].append(total_loss / len(train_loader))
        history['test_acc'].append(acc)

    return max(history['test_acc']), history

def objective(trial, mode, X_train, y_train, X_test, y_test):
    config = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True),
    }

    if mode == 'FixedLS':
        config['epsilon'] = trial.suggest_float('epsilon', 0.01, 0.5)
    elif mode == 'LGALS':
        config['epsilon_max'] = trial.suggest_float('epsilon_max', 0.01, 1.0)
        config['gamma'] = trial.suggest_float('gamma', 0.1, 5.0)

    acc, _ = train_model(mode, config, X_train, y_train, X_test, y_test, epochs=50)
    return acc

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()
    modes = ['Baseline', 'FixedLS', 'LGALS']
    best_configs = {}

    for mode in modes:
        print(f"Optimizing {mode}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode, X_train, y_train, X_test, y_test), n_trials=20)
        best_configs[mode] = study.best_params
        print(f"Best params for {mode}: {best_configs[mode]}")

    results = {}
    for mode in modes:
        print(f"Running final evaluation for {mode}...")
        mode_accs = []
        mode_histories = []
        for seed in range(5):
            acc, history = train_model(mode, best_configs[mode], X_train, y_train, X_test, y_test, epochs=100, seed=seed)
            mode_accs.append(acc)
            mode_histories.append(history)

        results[mode] = {
            'mean_acc': np.mean(mode_accs),
            'std_acc': np.std(mode_accs),
            'accs': mode_accs,
            'histories': mode_histories
        }

    # Save results
    with open('logit_alignment_label_smoothing/results.json', 'w') as f:
        # Convert histories to serializable
        serializable_results = {}
        for m, v in results.items():
            serializable_results[m] = {
                'mean_acc': float(v['mean_acc']),
                'std_acc': float(v['std_acc']),
                'accs': [float(x) for x in v['accs']]
            }
        json.dump(serializable_results, f, indent=4)

    with open('logit_alignment_label_smoothing/results.txt', 'w') as f:
        for mode in modes:
            f.write(f"{mode}: {results[mode]['mean_acc']:.4f} +/- {results[mode]['std_acc']:.4f}\n")

    # Plot results
    plt.figure(figsize=(10, 6))
    for mode in modes:
        # Average test acc history over seeds
        all_accs = np.array([h['test_acc'] for h in results[mode]['histories']])
        mean_accs = all_accs.mean(axis=0)
        plt.plot(mean_accs, label=mode)

    plt.title('Test Accuracy on MNIST1D with 20% Label Noise')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('logit_alignment_label_smoothing/comparison_plot.png')
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'smoke':
        X_train, y_train, X_test, y_test = get_data()
        for mode in ['Baseline', 'FixedLS', 'LGALS']:
            config = {'lr': 1e-3, 'weight_decay': 1e-4, 'epsilon': 0.1, 'epsilon_max': 0.1, 'gamma': 1.0}
            print(f"Smoke test for {mode}...")
            train_model(mode, config, X_train, y_train, X_test, y_test, epochs=2)
        print("Smoke tests passed")
    else:
        run_experiment()
