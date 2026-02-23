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

def compute_per_sample_grads(model, x, y):
    params = dict(model.named_parameters())
    param_names = list(params.keys())
    param_values = tuple(params.values())

    def loss_fn(p_values, x_single, y_single):
        p_dict = {name: val for name, val in zip(param_names, p_values)}
        logits = functional_call(model, p_dict, (x_single.unsqueeze(0),))
        return F.cross_entropy(logits, y_single.unsqueeze(0))

    grad_fn = grad(loss_fn)
    v_grad_fn = vmap(grad_fn, in_dims=(None, 0, 0))

    per_sample_grads = v_grad_fn(param_values, x, y)
    return per_sample_grads

def train_epoch(model, loader, optimizer, mode='baseline', clip_val=None, k=1.5):
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
        elif mode == 'global_clip':
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()
        elif mode in ['fixed_per_sample_clip', 'rogc']:
            per_sample_grads = compute_per_sample_grads(model, x, y)
            batch_size = x.size(0)

            # Compute norms
            norms = torch.zeros(batch_size, device=device)
            for g in per_sample_grads:
                norms += g.view(batch_size, -1).pow(2).sum(dim=1)
            norms = norms.sqrt()

            if mode == 'fixed_per_sample_clip':
                threshold = clip_val
            else: # rogc
                q1 = torch.quantile(norms, 0.25)
                q3 = torch.quantile(norms, 0.75)
                iqr = q3 - q1
                threshold = q3 + k * iqr

            scale = torch.clamp(threshold / (norms + 1e-8), max=1.0)

            for p, ps_grad in zip(model.parameters(), per_sample_grads):
                # ps_grad is [Batch, ...]
                # scale is [Batch]
                # we want to multiply them: scale.view(Batch, 1, 1...) * ps_grad
                expanded_scale = scale.view(batch_size, *([1] * (ps_grad.dim() - 1)))
                p.grad = (ps_grad * expanded_scale).mean(dim=0)

            optimizer.step()
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
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    clip_val = None
    k = 1.5
    if mode == 'global_clip' or mode == 'fixed_per_sample_clip':
        clip_val = trial.suggest_float('clip_val', 0.1, 10.0, log=True)
    elif mode == 'rogc':
        k = trial.suggest_float('k', 0.0, 5.0)

    best_val_acc = 0
    for epoch in range(15):
        train_epoch(model, train_loader, optimizer, mode=mode, clip_val=clip_val, k=k)
        val_acc = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'global_clip', 'fixed_per_sample_clip', 'rogc', 'all'])
    parser.add_argument('--data', type=str, default='clean', choices=['clean', 'noisy'])
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--seeds', type=str, default='42')
    args = parser.parse_args()

    if args.smoke_test:
        X_train, y_train, X_val, y_val, X_test, y_test = get_data(noisy_fraction=0.0)
        train_loader = TensorDataLoader((X_train[:128], y_train[:128]), batch_size=32, shuffle=True)
        model = MLP().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        for mode in ['baseline', 'global_clip', 'fixed_per_sample_clip', 'rogc']:
            print(f"Smoke test for {mode}...")
            train_epoch(model, train_loader, optimizer, mode=mode, clip_val=1.0, k=1.5)
        print("Smoke test passed!")
        return

    noisy_fraction = 0.2 if args.data == 'noisy' else 0.0
    X_train, y_train, X_val, y_val, X_test, y_test = get_data(noisy_fraction=noisy_fraction)
    data_config = (X_train, y_train, X_val, y_val)

    best_params_file = f'robust_outlier_clipping_experiment/best_params_{args.data}.json'
    if os.path.exists(best_params_file):
        with open(best_params_file, 'r') as f:
            best_params_all = json.load(f)
    else:
        best_params_all = {}

    if args.tune:
        print(f"Tuning {args.mode} on {args.data} data...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, args.mode, data_config), n_trials=20)
        best_params_all[args.mode] = study.best_params
        with open(best_params_file, 'w') as f:
            json.dump(best_params_all, f, indent=4)
        print(f"Best params for {args.mode}: {study.best_params}")

    if args.evaluate:
        seeds = [int(s) for s in args.seeds.split(',')]
        modes = ['baseline', 'global_clip', 'fixed_per_sample_clip', 'rogc'] if args.mode == 'all' else [args.mode]

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
                clip_val = params.get('clip_val')
                k = params.get('k', 1.5)

                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

                losses = []
                accs = []
                for epoch in range(30):
                    loss = train_epoch(model, train_loader, optimizer, mode=mode, clip_val=clip_val, k=k)
                    val_acc = evaluate(model, val_loader)
                    losses.append(loss)
                    accs.append(val_acc)

                test_acc = evaluate(model, test_loader)
                mode_test_accs.append(test_acc)
                mode_train_losses.append(losses)
                mode_val_accs.append(accs)

            results[mode] = {
                'test_acc_mean': np.mean(mode_test_accs),
                'test_acc_std': np.std(mode_test_accs),
                'train_loss': np.mean(mode_train_losses, axis=0).tolist(),
                'val_acc': np.mean(mode_val_accs, axis=0).tolist()
            }

        # Save results
        with open(f'robust_outlier_clipping_experiment/results_{args.data}.json', 'w') as f:
            json.dump(results, f, indent=4)

        # Plotting
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for mode in results:
            plt.plot(results[mode]['train_loss'], label=mode)
        plt.title(f'Train Loss ({args.data})')
        plt.legend()

        plt.subplot(1, 2, 2)
        for mode in results:
            plt.plot(results[mode]['val_acc'], label=mode)
        plt.title(f'Val Accuracy ({args.data})')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'robust_outlier_clipping_experiment/plots_{args.data}.png')
        plt.close()

if __name__ == '__main__':
    main()
