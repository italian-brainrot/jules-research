import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import vmap, grad, functional_call
import optuna
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import argparse
import json
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data(noisy_fraction=0.0):
    args = get_dataset_args()
    args.num_samples = 4000
    data = make_dataset(args)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    if noisy_fraction > 0:
        n_noisy = int(noisy_fraction * len(y_train))
        noisy_indices = torch.randperm(len(y_train))[:n_noisy]
        # Randomize labels
        y_train[noisy_indices] = torch.randint(0, 10, (n_noisy,))

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

def compute_gnvr_grads(model, x, y, lambd):
    params = dict(model.named_parameters())
    param_names = list(params.keys())
    param_values = tuple(params.values())
    buffers = dict(model.named_buffers())

    def loss_fn(p_values, b, x_single, y_single):
        p_dict = {name: val for name, val in zip(param_names, p_values)}
        sd = {**p_dict, **b}
        logits = functional_call(model, sd, (x_single.unsqueeze(0),))
        return F.cross_entropy(logits, y_single.unsqueeze(0))

    def per_sample_grad_norm(p_values, b, x_single, y_single):
        g = grad(loss_fn)(p_values, b, x_single, y_single)
        norm_sq = sum(p_g.pow(2).sum() for p_g in g)
        return torch.sqrt(norm_sq + 1e-8)

    def total_loss_functional(p_values, b, xb, yb, l):
        # Mean task loss
        task_losses = vmap(loss_fn, in_dims=(None, None, 0, 0))(p_values, b, xb, yb)
        mean_task_loss = task_losses.mean()

        # GNVR loss: variance of per-sample grad norms
        norms = vmap(per_sample_grad_norm, in_dims=(None, None, 0, 0))(p_values, b, xb, yb)
        gnvr_loss = torch.var(norms)

        return mean_task_loss + l * gnvr_loss

    # Gradient of total_loss_functional w.r.t p_values
    grads = grad(total_loss_functional)(param_values, buffers, x, y, lambd)
    return grads

def train_epoch(model, loader, optimizer, mode='baseline', lambd=0.0):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if mode == 'baseline':
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
        elif mode == 'gnvr':
            grads = compute_gnvr_grads(model, x, y, lambd)
            for p, g in zip(model.parameters(), grads):
                p.grad = g
            optimizer.step()
            # For logging
            with torch.no_grad():
                output = model(x)
                loss = F.cross_entropy(output, y)

        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def objective(trial, mode, data_config):
    set_seed(42)
    X_train, y_train, X_val, y_val = data_config
    train_loader = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    val_loader = TensorDataLoader((X_val, y_val), batch_size=64, shuffle=False)

    model = MLP().to(device)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)

    lambd = 0.0
    if mode == 'gnvr':
        lambd = trial.suggest_float('lambd', 1e-3, 1e1, log=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_val_acc = 0
    for epoch in range(15):
        train_epoch(model, train_loader, optimizer, mode=mode, lambd=lambd)
        val_acc = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'gnvr', 'all'])
    parser.add_argument('--data', type=str, default='clean', choices=['clean', 'noisy'])
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--seeds', type=str, default='42')
    args = parser.parse_args()

    noisy_fraction = 0.2 if args.data == 'noisy' else 0.0
    X_train, y_train, X_val, y_val, X_test, y_test = get_data(noisy_fraction=noisy_fraction)
    data_config = (X_train, y_train, X_val, y_val)

    best_params_file = f'gradient_norm_variance_regularization/best_params_{args.data}.json'
    if os.path.exists(best_params_file):
        with open(best_params_file, 'r') as f:
            best_params_all = json.load(f)
    else:
        best_params_all = {}

    modes = ['baseline', 'gnvr'] if args.mode == 'all' else [args.mode]

    if args.tune:
        for mode in modes:
            print(f"Tuning {mode} on {args.data} data...")
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, mode, data_config), n_trials=20)
            best_params_all[mode] = study.best_params
            with open(best_params_file, 'w') as f:
                json.dump(best_params_all, f, indent=4)
            print(f"Best params for {mode}: {study.best_params}")

    if args.evaluate:
        seeds = [int(s) for s in args.seeds.split(',')]

        results = {}
        train_loader = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
        val_loader = TensorDataLoader((X_val, y_val), batch_size=64, shuffle=False)
        test_loader = TensorDataLoader((X_test, y_test), batch_size=64, shuffle=False)

        for mode in modes:
            if mode not in best_params_all:
                print(f"Skipping {mode} as no best params found.")
                continue

            mode_test_accs = []
            mode_train_losses = []
            mode_val_accs = []

            for seed in seeds:
                print(f"Evaluating {mode} with seed {seed}...")
                set_seed(seed)
                model = MLP().to(device)
                params = best_params_all[mode]
                lr = params['lr']
                wd = params['weight_decay']
                lambd = params.get('lambd', 0.0)

                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

                losses = []
                accs = []
                for epoch in range(30):
                    loss = train_epoch(model, train_loader, optimizer, mode=mode, lambd=lambd)
                    val_acc = evaluate(model, val_loader)
                    losses.append(loss)
                    accs.append(val_acc)
                    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")

                test_acc = evaluate(model, test_loader)
                mode_test_accs.append(test_acc)
                mode_train_losses.append(losses)
                mode_val_accs.append(accs)
                print(f"Seed {seed} Test Acc: {test_acc:.4f}")

            results[mode] = {
                'test_acc_mean': np.mean(mode_test_accs),
                'test_acc_std': np.std(mode_test_accs),
                'train_loss': np.mean(mode_train_losses, axis=0).tolist(),
                'val_acc': np.mean(mode_val_accs, axis=0).tolist()
            }

        # Save results
        with open(f'gradient_norm_variance_regularization/results_{args.data}.json', 'w') as f:
            json.dump(results, f, indent=4)

        # Plotting
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for mode in results:
            plt.plot(results[mode]['train_loss'], label=mode)
        plt.title(f'Train Loss ({args.data})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        for mode in results:
            plt.plot(results[mode]['val_acc'], label=mode)
        plt.title(f'Val Accuracy ({args.data})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'gradient_norm_variance_regularization/plots_{args.data}.png')
        plt.close()
        print(f"Results: {results}")

if __name__ == '__main__':
    main()
