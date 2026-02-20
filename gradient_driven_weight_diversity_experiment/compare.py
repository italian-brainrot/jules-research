import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mnist1d
import optuna
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
import os
import random

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
    data = mnist1d.get_dataset(args, path='gradient_driven_weight_diversity_experiment/mnist1d_data.pkl', download=True)
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
    def __init__(self, input_size=40, hidden_size=512, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.activations = {}

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        self.activations['fc1'] = x1
        x2 = F.relu(self.fc2(x1))
        self.activations['fc2'] = x2
        x3 = self.fc3(x2)
        return x3

def get_gwd_penalty(model, alpha, mode, optimizer, gamma=1.0):
    if mode == 'baseline' or alpha == 0:
        return torch.tensor(0.0, device=device)

    penalty = 0
    count = 0
    for name in ['fc1', 'fc2']:
        module = getattr(model, name)
        a = model.activations.get(name)
        w = module.weight

        if a is None:
            continue

        # Get moving average gradient from optimizer
        if w in optimizer.state and 'exp_avg' in optimizer.state[w]:
            g = optimizer.state[w]['exp_avg']
        else:
            g = module.weight.grad

        if g is None:
            continue

        # Compute correlation of activations
        a_centered = a - a.mean(dim=0, keepdim=True)
        cov = torch.mm(a_centered.t(), a_centered) / (a.size(0) - 1)
        std = torch.sqrt(torch.diag(cov) + 1e-8)
        corr = cov / torch.outer(std, std)

        mask = torch.triu(torch.ones_like(corr), diagonal=1)

        if mode == 'gwd':
            g_norm = F.normalize(g.detach(), p=2, dim=1)
            sim_g = torch.mm(g_norm, g_norm.t())
            p = (corr ** 2) * torch.relu(sim_g) ** gamma * mask
        elif mode == 'decorr':
            p = (corr ** 2) * mask
        else:
            continue

        penalty += p.sum()
        count += mask.sum()

    if count == 0:
        return torch.tensor(0.0, device=device)
    return alpha * (penalty / count)

def train_epoch(model, loader, optimizer, criterion, alpha, mode, gamma=1.0):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)

        # We need gradients for gwd if not using exp_avg, but even with exp_avg
        # it's better to have current gradients too.
        # Actually, let's just do one backward.
        loss.backward(retain_graph=(mode != 'baseline'))

        if mode != 'baseline':
            penalty = get_gwd_penalty(model, alpha, mode, optimizer, gamma)
            if penalty.item() > 0:
                penalty.backward()

        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

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
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)

    model = MLP().to(device)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)

    alpha = 0
    gamma = 1.0
    if mode != 'baseline':
        alpha = trial.suggest_float('alpha', 1e-3, 10.0, log=True) # Increased range because of mean
        if mode == 'gwd':
            gamma = trial.suggest_float('gamma', 0.1, 3.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(25):
        train_epoch(model, train_loader, optimizer, criterion, alpha, mode, gamma)
        _, val_acc = evaluate(model, val_loader, criterion)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_acc

def run_experiment():
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)
    test_loader = TensorDataLoader((x_test, y_test), batch_size=128, shuffle=False)

    modes = ['baseline', 'decorr', 'gwd']
    best_params = {}

    data_for_obj = (x_train, y_train, x_val, y_val)
    for mode in modes:
        print(f"Optimizing {mode}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode, data_for_obj), n_trials=30)
        best_params[mode] = study.best_params
        print(f"Best params for {mode}: {study.best_params}")

    # Final training
    results = {}
    for mode in modes:
        print(f"Final training for {mode}...")
        set_seed(42)
        model = MLP().to(device)
        params = best_params[mode].copy()
        lr = params.pop('lr')
        wd = params.pop('weight_decay')
        alpha = params.pop('alpha', 0)
        gamma = params.pop('gamma', 1.0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        val_accs = []
        best_model_state = None
        best_val_acc = 0

        for epoch in range(100):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, alpha, mode, gamma)
            _, val_acc = evaluate(model, val_loader, criterion)
            train_losses.append(train_loss)
            val_accs.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

        model.load_state_dict(best_model_state)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        results[mode] = {
            'train_losses': train_losses,
            'val_accs': val_accs,
            'test_acc': test_acc,
            'best_val_acc': best_val_acc
        }
        print(f"{mode} Test Acc: {test_acc:.4f} (Best Val Acc: {best_val_acc:.4f})")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for mode in modes:
        plt.plot(results[mode]['train_losses'], label=mode)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for mode in modes:
        plt.plot(results[mode]['val_accs'], label=mode)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('gradient_driven_weight_diversity_experiment/results.png')

    # Save text results
    with open('gradient_driven_weight_diversity_experiment/results.txt', 'w') as f:
        for mode in modes:
            f.write(f"{mode} Best Params: {best_params[mode]}\n")
            f.write(f"{mode} Test Acc: {results[mode]['test_acc']:.4f}\n")
            f.write(f"{mode} Best Val Acc: {results[mode]['best_val_acc']:.4f}\n")

if __name__ == '__main__':
    run_experiment()
