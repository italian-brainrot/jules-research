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
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from optimizer import GAM

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data():
    args = get_dataset_args()
    data = make_dataset(args)
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

def compute_gsnr_and_grads(model, x, y):
    params = dict(model.named_parameters())
    param_names = list(params.keys())
    param_values = tuple(params.values())

    def loss_fn(p_values, x_single, y_single):
        p_dict = {name: val for name, val in zip(param_names, p_values)}
        # functional_call expects x_single to have batch dim for the model,
        # but since we vmap, it will have it if we handle it right.
        # Actually x_single will be [40], we need [1, 40] for the model.
        logits = functional_call(model, p_dict, (x_single.unsqueeze(0),))
        return F.cross_entropy(logits, y_single.unsqueeze(0))

    grad_fn = grad(loss_fn)
    v_grad_fn = vmap(grad_fn, in_dims=(None, 0, 0))

    per_sample_grads = v_grad_fn(param_values, x, y)

    gsnrs = {}
    batch_grads = []

    # per_sample_grads is a tuple of tensors, one for each parameter
    for p, ps_grad in zip(model.parameters(), per_sample_grads):
        mean_g = ps_grad.mean(dim=0)
        sq_g = (ps_grad**2).mean(dim=0)
        gsnr = (mean_g**2) / (sq_g + 1e-8)
        gsnrs[p] = gsnr
        batch_grads.append(mean_g)

    return gsnrs, batch_grads

def train_epoch(model, loader, optimizer, use_gam=False):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if use_gam:
            gsnrs, batch_grads = compute_gsnr_and_grads(model, x, y)
            for p, g in zip(model.parameters(), batch_grads):
                p.grad = g
            optimizer.step(gsnrs=gsnrs)
            # Compute loss for tracking
            with torch.no_grad():
                output = model(x)
                loss = F.cross_entropy(output, y)
        else:
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

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

def objective(trial, opt_name, data):
    set_seed(42)
    x_train, y_train, x_val, y_val = data
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)

    model = MLP().to(device)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    if opt_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        gamma = trial.suggest_float('gamma', 0.1, 5.0)
        optimizer = GAM(model.parameters(), lr=lr, weight_decay=wd, gamma=gamma)

    best_val_acc = 0
    for epoch in range(15):
        train_epoch(model, train_loader, optimizer, use_gam=(opt_name == 'GAM'))
        val_acc = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def run_experiment():
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)
    test_loader = TensorDataLoader((x_test, y_test), batch_size=128, shuffle=False)

    optimizers = ['AdamW', 'GAM']
    best_params = {}

    data_for_obj = (x_train, y_train, x_val, y_val)
    for opt_name in optimizers:
        print(f"Optimizing {opt_name}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, opt_name, data_for_obj), n_trials=15)
        best_params[opt_name] = study.best_params
        print(f"Best params for {opt_name}: {study.best_params}")

    # Final training with multiple seeds
    final_results = {}
    seeds = [42, 43, 44]

    for opt_name in optimizers:
        opt_test_accs = []
        all_train_losses = []
        all_val_accs = []

        for seed in seeds:
            print(f"Final training for {opt_name} (seed {seed})...")
            set_seed(seed)
            model = MLP().to(device)
            params = best_params[opt_name]
            if opt_name == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), **params)
            else:
                optimizer = GAM(model.parameters(), **params)

            train_losses = []
            val_accs = []

            for epoch in range(30):
                train_loss = train_epoch(model, train_loader, optimizer, use_gam=(opt_name == 'GAM'))
                val_acc = evaluate(model, val_loader)
                train_losses.append(train_loss)
                val_accs.append(val_acc)

            test_acc = evaluate(model, test_loader)
            opt_test_accs.append(test_acc)
            all_train_losses.append(train_losses)
            all_val_accs.append(val_accs)
            print(f"Seed {seed} Test Acc: {test_acc:.4f}")

        final_results[opt_name] = {
            'test_acc_mean': np.mean(opt_test_accs),
            'test_acc_std': np.std(opt_test_accs),
            'train_losses': np.mean(all_train_losses, axis=0),
            'val_accs': np.mean(all_val_accs, axis=0)
        }

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for opt_name in optimizers:
        plt.plot(final_results[opt_name]['train_losses'], label=opt_name)
    plt.title('Average Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for opt_name in optimizers:
        plt.plot(final_results[opt_name]['val_accs'], label=opt_name)
    plt.title('Average Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('gsnr_adaptive_momentum_experiment/results.png')

    # Save text results
    with open('gsnr_adaptive_momentum_experiment/results.txt', 'w') as f:
        for opt_name in optimizers:
            f.write(f"{opt_name} Best Params: {best_params[opt_name]}\n")
            f.write(f"{opt_name} Test Acc: {final_results[opt_name]['test_acc_mean']:.4f} +- {final_results[opt_name]['test_acc_std']:.4f}\n")

if __name__ == '__main__':
    run_experiment()
