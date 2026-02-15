import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call
import numpy as np
import mnist1d
import optuna
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
import os
import random
import copy

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data():
    args = mnist1d.get_dataset_args()
    data = mnist1d.get_dataset(args, path='gsnr_adaptive_accumulation_experiment/mnist1d_data.pkl', download=True)
    x_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    x_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    # Split train into train and val
    n_train = int(0.8 * len(x_train))
    x_val = x_train[n_train:]
    y_val = y_train[n_train:]
    x_train = x_train[:n_train]
    y_train = y_train[:n_train]

    return x_train, y_train, x_val, y_val, x_test, y_test

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

def compute_loss(params, buffers, x, y, model, criterion):
    # Functional call of the model
    logits = functional_call(model, (params, buffers), (x.unsqueeze(0),))
    return criterion(logits, y.unsqueeze(0))

def get_per_sample_grads(model, x, y, criterion):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    # We want gradients w.r.t params
    def loss_fn(p, b, x_single, y_single):
        logits = functional_call(model, (p, b), (x_single.unsqueeze(0),))
        return criterion(logits, y_single.unsqueeze(0))

    grad_fn = grad(loss_fn)
    per_sample_grads = vmap(grad_fn, in_dims=(None, None, 0, 0))(params, buffers, x, y)
    return per_sample_grads

def train_baseline(model, loader, optimizer, criterion, total_samples_limit):
    model.train()
    samples_used = 0
    total_loss = 0
    steps = 0

    loader_iter = iter(loader)
    while samples_used < total_samples_limit:
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        samples_used += x.size(0)
        total_loss += loss.item()
        steps += 1

    return total_loss / steps, samples_used

def train_gaga(model, loader, optimizer, criterion, total_samples_limit, tau, k_min, k_max):
    model.train()
    samples_used = 0
    total_loss = 0
    steps = 0

    loader_iter = iter(loader)

    while samples_used < total_samples_limit:
        k = 0
        s1 = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
        s2 = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

        batch_losses = []

        while k < k_max:
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)

            x, y = x.to(device), y.to(device)

            # Compute per-sample grads
            ps_grads = get_per_sample_grads(model, x, y, criterion)

            batch_k = x.size(0)
            k += batch_k

            for name, g in ps_grads.items():
                s1[name] += g.sum(dim=0)
                s2[name] += (g**2).sum(dim=0)

            # Compute loss for reporting
            with torch.no_grad():
                logits = model(x)
                loss = criterion(logits, y)
                batch_losses.append(loss.item())

            if k >= k_min:
                # Calculate GSNR
                gsnr_values = []
                for name in s1:
                    m = s1[name] / k
                    v = (s2[name] / k) - (m**2)
                    v = torch.clamp(v, min=0.0)
                    gsnr_p = (m**2) / (v / k + 1e-8)
                    gsnr_values.append(gsnr_p.mean())

                avg_gsnr = torch.stack(gsnr_values).mean().item()
                if avg_gsnr > tau:
                    break

        # Apply gradients
        optimizer.zero_grad()
        for name, p in model.named_parameters():
            p.grad = s1[name] / k
        optimizer.step()

        samples_used += k
        total_loss += sum(batch_losses) / len(batch_losses)
        steps += 1

    return total_loss / steps, samples_used

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    return total_loss / len(loader), correct / loader.data_length()

def objective(trial, mode, data):
    set_seed(42)
    x_train, y_train, x_val, y_val = data
    # Small batch size for GAGA increments, and for baseline for comparison
    batch_size = 32
    train_loader = TensorDataLoader((x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)

    model = MLP().to(device)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_samples_per_epoch = len(x_train)
    total_budget = total_samples_per_epoch * 10 # 10 "epochs" worth of samples

    if mode == 'baseline':
        train_baseline(model, train_loader, optimizer, criterion, total_budget)
    else:
        tau = trial.suggest_float('tau', 0.01, 1.0, log=True)
        train_gaga(model, train_loader, optimizer, criterion, total_budget, tau=tau, k_min=32, k_max=512)

    _, val_acc = evaluate(model, val_loader, criterion)
    return val_acc

def main():
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    data_for_obj = (x_train, y_train, x_val, y_val)

    print("Optimizing Baseline...")
    study_baseline = optuna.create_study(direction='maximize')
    study_baseline.optimize(lambda trial: objective(trial, 'baseline', data_for_obj), n_trials=20)
    best_lr_baseline = study_baseline.best_params['lr']
    print(f"Best Baseline LR: {best_lr_baseline}")

    print("Optimizing GAGA...")
    study_gaga = optuna.create_study(direction='maximize')
    study_gaga.optimize(lambda trial: objective(trial, 'gaga', data_for_obj), n_trials=20)
    best_params_gaga = study_gaga.best_params
    print(f"Best GAGA Params: {best_params_gaga}")

    # Final Evaluation
    set_seed(42)
    model_baseline = MLP().to(device)
    opt_baseline = torch.optim.Adam(model_baseline.parameters(), lr=best_lr_baseline)

    set_seed(42)
    model_gaga = MLP().to(device)
    opt_gaga = torch.optim.Adam(model_gaga.parameters(), lr=best_params_gaga['lr'])

    criterion = nn.CrossEntropyLoss()

    batch_size = 32
    train_loader = TensorDataLoader((x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)
    test_loader = TensorDataLoader((x_test, y_test), batch_size=128, shuffle=False)

    total_samples_per_epoch = len(x_train)
    num_epochs = 30
    total_budget_per_epoch = total_samples_per_epoch

    results = {'baseline': {'acc': [], 'samples': []}, 'gaga': {'acc': [], 'samples': [], 'k_avg': []}}

    print("\nFinal Training Baseline...")
    samples_used_total = 0
    for epoch in range(num_epochs):
        loss, samples = train_baseline(model_baseline, train_loader, opt_baseline, criterion, total_budget_per_epoch)
        samples_used_total += samples
        _, val_acc = evaluate(model_baseline, val_loader, criterion)
        results['baseline']['acc'].append(val_acc)
        results['baseline']['samples'].append(samples_used_total)
        print(f"Epoch {epoch+1}, Samples: {samples_used_total}, Val Acc: {val_acc:.4f}")

    print("\nFinal Training GAGA...")
    samples_used_total = 0
    for epoch in range(num_epochs):
        # We want to use roughly the same amount of samples per "epoch"
        # But GAGA's k is variable. We just run it for total_budget_per_epoch samples.
        loss, samples = train_gaga(model_gaga, train_loader, opt_gaga, criterion, total_budget_per_epoch,
                                   tau=best_params_gaga['tau'], k_min=32, k_max=512)
        samples_used_total += samples
        _, val_acc = evaluate(model_gaga, val_loader, criterion)
        results['gaga']['acc'].append(val_acc)
        results['gaga']['samples'].append(samples_used_total)
        print(f"Epoch {epoch+1}, Samples: {samples_used_total}, Val Acc: {val_acc:.4f}")

    # Test set results
    _, test_acc_baseline = evaluate(model_baseline, test_loader, criterion)
    _, test_acc_gaga = evaluate(model_gaga, test_loader, criterion)

    print(f"\nFinal Test Acc Baseline: {test_acc_baseline:.4f}")
    print(f"Final Test Acc GAGA: {test_acc_gaga:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(results['baseline']['samples'], results['baseline']['acc'], label='Baseline (Fixed BS=32)')
    plt.plot(results['gaga']['samples'], results['gaga']['acc'], label=f'GAGA (Tau={best_params_gaga["tau"]:.4f})')
    plt.xlabel('Total Samples Used')
    plt.ylabel('Validation Accuracy')
    plt.title('GAGA vs Baseline Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('gsnr_adaptive_accumulation_experiment/comparison.png')

    # Save results to file
    with open('gsnr_adaptive_accumulation_experiment/results.txt', 'w') as f:
        f.write(f"Best Baseline LR: {best_lr_baseline}\n")
        f.write(f"Best GAGA Params: {best_params_gaga}\n")
        f.write(f"Test Acc Baseline: {test_acc_baseline}\n")
        f.write(f"Test Acc GAGA: {test_acc_gaga}\n")

if __name__ == "__main__":
    main()
