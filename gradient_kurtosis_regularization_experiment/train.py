import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
import optuna
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from gkr_lib import get_gkr_loss, compute_per_sample_grad_norms, compute_gradient_kurtosis

# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 4000
    data = make_dataset(defaults)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    # Validation split
    n_val = 500
    X_val = X_train[-n_val:]
    y_val = y_train[-n_val:]
    X_train = X_train[:-n_val]
    y_train = y_train[:-n_val]

    return X_train, y_train, X_val, y_val, X_test, y_test

class MLP(nn.Module):
    def __init__(self, input_size=40, hidden_sizes=[128, 128], num_classes=10):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            in_size = h_size
        layers.append(nn.Linear(in_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        preds = outputs.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc

def train_epoch(model, dl, optimizer, lambda_gkr, device):
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_gkr_loss = 0
    total_kurtosis = 0

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # CE loss
        outputs = model(x)
        ce_loss = F.cross_entropy(outputs, y)

        # GKR loss
        if lambda_gkr > 0:
            # We need to compute GKR loss. gkr_lib uses functional_call
            gkr_loss = get_gkr_loss(model, x, y, lambda_gkr)
            loss = ce_loss + gkr_loss

            with torch.no_grad():
                norms = compute_per_sample_grad_norms(model, x, y)
                kurt = compute_gradient_kurtosis(norms).item()
        else:
            loss = ce_loss
            gkr_loss = torch.tensor(0.0, device=device)
            with torch.no_grad():
                norms = compute_per_sample_grad_norms(model, x, y)
                kurt = compute_gradient_kurtosis(norms).item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_gkr_loss += gkr_loss.item()
        total_kurtosis += kurt

    n = len(dl)
    return total_loss/n, total_ce_loss/n, total_gkr_loss/n, total_kurtosis/n

def objective(trial, X_train, y_train, X_val, y_val, use_gkr, device):
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    lambda_gkr = 0.0
    if use_gkr:
        lambda_gkr = trial.suggest_float('lambda_gkr', 1e-6, 1.0, log=True)

    set_seed(42)
    model = MLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

    epochs = 15
    for epoch in range(epochs):
        train_epoch(model, dl_train, optimizer, lambda_gkr, device)

    return evaluate(model, X_val, y_val, device)

def run_final_training(X_train, y_train, X_test, y_test, lr, lambda_gkr, device, epochs=50):
    set_seed(42)
    model = MLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

    history = {
        'loss': [],
        'ce_loss': [],
        'gkr_loss': [],
        'kurtosis': [],
        'test_acc': []
    }

    for epoch in range(epochs):
        loss, ce, gkr, kurt = train_epoch(model, dl_train, optimizer, lambda_gkr, device)
        acc = evaluate(model, X_test, y_test, device)

        history['loss'].append(loss)
        history['ce_loss'].append(ce)
        history['gkr_loss'].append(gkr)
        history['kurtosis'].append(kurt)
        history['test_acc'].append(acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss {loss:.4f}, Kurtosis {kurt:.4f}, Test Acc {acc:.4f}")

    return history

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train, X_val, y_val, X_test, y_test = get_data()

    print("Tuning Baseline...")
    study_baseline = optuna.create_study(direction='maximize')
    study_baseline.optimize(lambda t: objective(t, X_train, y_train, X_val, y_val, False, device), n_trials=10)
    best_lr_baseline = study_baseline.best_params['lr']

    print(f"Best Baseline LR: {best_lr_baseline}")

    print("\nTuning GKR...")
    study_gkr = optuna.create_study(direction='maximize')
    study_gkr.optimize(lambda t: objective(t, X_train, y_train, X_val, y_val, True, device), n_trials=15)
    best_lr_gkr = study_gkr.best_params['lr']
    best_lambda_gkr = study_gkr.best_params['lambda_gkr']

    print(f"Best GKR LR: {best_lr_gkr}, Lambda: {best_lambda_gkr}")

    print("\nFinal Training Baseline...")
    hist_baseline = run_final_training(X_train, y_train, X_test, y_test, best_lr_baseline, 0.0, device)

    print("\nFinal Training GKR...")
    hist_gkr = run_final_training(X_train, y_train, X_test, y_test, best_lr_gkr, best_lambda_gkr, device)

    # Plotting
    epochs = len(hist_baseline['test_acc'])
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(hist_baseline['test_acc'], label='Baseline')
    plt.plot(hist_gkr['test_acc'], label='GKR')
    plt.title('Test Accuracy')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(hist_baseline['ce_loss'], label='Baseline CE')
    plt.plot(hist_gkr['ce_loss'], label='GKR CE')
    plt.title('Cross Entropy Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(hist_baseline['kurtosis'], label='Baseline')
    plt.plot(hist_gkr['kurtosis'], label='GKR')
    plt.title('Gradient Kurtosis')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(hist_gkr['gkr_loss'], label='GKR Loss')
    plt.title('GKR Regularization Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results.png')

    # Save text results
    with open('results.txt', 'w') as f:
        f.write(f"Baseline - Best LR: {best_lr_baseline}\n")
        f.write(f"Baseline - Final Test Acc: {hist_baseline['test_acc'][-1]}\n")
        f.write(f"GKR - Best LR: {best_lr_gkr}, Best Lambda: {best_lambda_gkr}\n")
        f.write(f"GKR - Final Test Acc: {hist_gkr['test_acc'][-1]}\n")

if __name__ == '__main__':
    main()
