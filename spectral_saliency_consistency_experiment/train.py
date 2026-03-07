import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call
import numpy as np
import optuna
import matplotlib.pyplot as plt
import os
import sys
import argparse
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from spectral_saliency_consistency_experiment.model import get_model

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_data():
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)

    x_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    x_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    # Val split
    n_val = 1000
    x_val = x_train[-n_val:]
    y_val = y_train[-n_val:]
    x_train = x_train[:-n_val]
    y_train = y_train[:-n_val]

    return x_train, y_train, x_val, y_val, x_test, y_test

def compute_ssc_loss(model, x, y, lambda_ssc):
    if lambda_ssc == 0:
        return torch.tensor(0.0, device=x.device)

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def compute_grads(params, buffers, x):
        params_and_buffers = {**params, **buffers}
        logits = functional_call(model, params_and_buffers, (x,))
        return logits

    # Compute Jacobian of logits w.r.t input
    # x: [B, D], logits: [B, C]
    # jac: [B, C, D]
    from torch.func import jacrev
    jac = vmap(jacrev(compute_grads, argnums=2), in_dims=(None, None, 0))(params, buffers, x)

    # Select the gradient for the target class
    # y: [B], jac: [B, C, D] -> [B, D]
    batch_idx = torch.arange(y.size(0), device=y.device)
    grads = jac[batch_idx, y]

    # Compute power spectrum of gradients
    # grads: [B, D]
    grads_fft = torch.fft.rfft(grads, dim=1)
    power_spectrum = torch.abs(grads_fft)

    # Normalize power spectrum for each sample to focus on "shape"
    # eps to avoid div by zero
    norm = torch.norm(power_spectrum, p=2, dim=1, keepdim=True) + 1e-8
    power_spectrum = power_spectrum / norm

    # Class-wise variance penalty
    total_var = 0.0
    num_classes_in_batch = 0
    unique_classes = torch.unique(y)

    for c in unique_classes:
        mask = (y == c)
        if mask.sum() > 1:
            class_spectra = power_spectrum[mask]
            # Variance across samples for each frequency component
            # [N_c, F] -> [F]
            var = torch.var(class_spectra, dim=0).mean()
            total_var += var
            num_classes_in_batch += 1

    if num_classes_in_batch > 0:
        return lambda_ssc * (total_var / num_classes_in_batch)
    else:
        return torch.tensor(0.0, device=x.device, requires_grad=True)

def train_epoch(model, loader, optimizer, criterion, mode, lambda_ssc, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        ce_loss = criterion(logits, y)

        if mode == 'ssc':
            ssc_loss = compute_ssc_loss(model, x, y, lambda_ssc)
            loss = ce_loss + ssc_loss
        else:
            loss = ce_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def objective(trial, mode, data, device):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    lambda_ssc = 0
    if mode == 'ssc':
        lambda_ssc = trial.suggest_float('lambda_ssc', 1e-4, 1.0, log=True)

    x_train, y_train, x_val, y_val = data
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)

    model = get_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(15):
        train_epoch(model, train_loader, optimizer, criterion, mode, lambda_ssc, device)
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['baseline', 'ssc', 'both'], default='both')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    data_for_tune = (x_train, y_train, x_val, y_val)

    best_params = {}

    modes = ['baseline', 'ssc'] if args.mode == 'both' else [args.mode]

    if args.tune:
        if os.path.exists('spectral_saliency_consistency_experiment/best_params.json'):
            with open('spectral_saliency_consistency_experiment/best_params.json', 'r') as f:
                import json
                best_params = json.load(f)
        else:
            best_params = {}

        for mode in modes:
            print(f"Tuning {mode}...")
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, mode, data_for_tune, device), n_trials=30)
            best_params[mode] = study.best_params
            print(f"Best params for {mode}: {best_params[mode]}")

        with open('spectral_saliency_consistency_experiment/best_params.json', 'w') as f:
            import json
            json.dump(best_params, f)
    else:
        if os.path.exists('spectral_saliency_consistency_experiment/best_params.json'):
            with open('spectral_saliency_consistency_experiment/best_params.json', 'r') as f:
                import json
                best_params = json.load(f)
        else:
            # Fallback if tune not run
            best_params = {
                'baseline': {'lr': 1e-3, 'weight_decay': 1e-4},
                'ssc': {'lr': 1e-3, 'weight_decay': 1e-4, 'lambda_ssc': 0.1}
            }

    if args.evaluate:
        results = {}
        histories = {}

        train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
        test_loader = TensorDataLoader((x_test, y_test), batch_size=128, shuffle=False)

        for mode in modes:
            print(f"Evaluating {mode}...")
            mode_accs = []
            mode_histories = []

            for seed in range(5):
                print(f"Seed {seed}")
                set_seed(seed)
                model = get_model().to(device)
                lr = best_params[mode]['lr']
                wd = best_params[mode]['weight_decay']
                lambda_ssc = best_params[mode].get('lambda_ssc', 0)

                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                criterion = nn.CrossEntropyLoss()

                history = {'train_loss': [], 'test_acc': []}
                for epoch in range(30):
                    loss = train_epoch(model, train_loader, optimizer, criterion, mode, lambda_ssc, device)
                    test_acc = evaluate(model, test_loader, device)
                    history['train_loss'].append(loss)
                    history['test_acc'].append(test_acc)

                mode_accs.append(test_acc)
                mode_histories.append(history)

            results[mode] = {
                'mean': np.mean(mode_accs),
                'std': np.std(mode_accs),
                'all': mode_accs
            }
            histories[mode] = mode_histories

        print("\nFinal Results:")
        for mode in modes:
            print(f"{mode}: {results[mode]['mean']:.4f} +/- {results[mode]['std']:.4f}")

        # Plotting
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for mode in modes:
            all_losses = np.array([h['train_loss'] for h in histories[mode]])
            plt.plot(all_losses.mean(axis=0), label=mode)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        for mode in modes:
            all_accs = np.array([h['test_acc'] for h in histories[mode]])
            plt.plot(all_accs.mean(axis=0), label=mode)
        plt.title('Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('spectral_saliency_consistency_experiment/results.png')

        with open('spectral_saliency_consistency_experiment/results.txt', 'w') as f:
            f.write("Spectral Saliency Consistency Experiment Results\n")
            f.write("==============================================\n")
            for mode in modes:
                f.write(f"\nMode: {mode}\n")
                f.write(f"Best Params: {best_params[mode]}\n")
                f.write(f"Test Accuracy: {results[mode]['mean']:.4f} +/- {results[mode]['std']:.4f}\n")
                f.write(f"All seeds: {results[mode]['all']}\n")

if __name__ == '__main__':
    main()
