import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import numpy as np
import optuna
import matplotlib.pyplot as plt
import os
import json

def get_data(num_samples=10000):
    defaults = get_dataset_args()
    defaults.num_samples = num_samples
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

class HomotopyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        nn.init.kaiming_normal_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, lambda_val):
        return (1 - lambda_val) * x + lambda_val * torch.relu(self.lin(x))

class HomotopyMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=64, num_layers=32, output_dim=10):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([HomotopyLayer(hidden_dim) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lambda_val):
        x = torch.relu(self.in_proj(x))
        for layer in self.layers:
            x = layer(x, lambda_val)
        return self.out_proj(x)

def train_model(config, X_train, y_train, X_test, y_test, epochs=100, trial=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HomotopyMLP(hidden_dim=config.get('hidden_dim', 64), num_layers=config.get('num_layers', 32)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)

    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        if config['variant'] == 'homotopy':
            T_warmup = int(epochs * config.get('warmup_fraction', 0.5))
            if epoch < T_warmup:
                lambda_val = epoch / T_warmup
            else:
                lambda_val = 1.0
        else:
            lambda_val = 1.0

        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx, lambda_val)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test, 1.0)
            test_acc = (test_logits.argmax(1) == y_test).float().mean().item()

        history['train_loss'].append(epoch_loss / len(train_loader))
        history['test_acc'].append(test_acc)

        if trial:
            trial.report(test_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return history, history['test_acc'][-1]

def objective(trial, variant, X_train, y_train, X_test, y_test, epochs=30):
    config = {
        'variant': variant,
        'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
    }
    if variant == 'homotopy':
        config['warmup_fraction'] = trial.suggest_float('warmup_fraction', 0.1, 0.9)
    _, final_acc = train_model(config, X_train, y_train, X_test, y_test, epochs=epochs, trial=trial)
    return final_acc

def run_experiment(n_trials=5, epochs_tune=30, epochs_final=100):
    X_train, y_train, X_test, y_test = get_data()
    variants = ['baseline', 'homotopy']
    best_configs = {}
    for variant in variants:
        print(f"Tuning {variant}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, variant, X_train, y_train, X_test, y_test, epochs=epochs_tune), n_trials=n_trials)
        best_configs[variant] = study.best_params
        best_configs[variant]['variant'] = variant
        print(f"Best params for {variant}: {study.best_params}")
    results = {}
    for variant in variants:
        print(f"Final training for {variant}...")
        history, final_acc = train_model(best_configs[variant], X_train, y_train, X_test, y_test, epochs=epochs_final)
        results[variant] = {'history': history, 'final_acc': final_acc, 'config': best_configs[variant]}
        print(f"Final accuracy for {variant}: {final_acc:.4f}")
    with open('residual_homotopy_experiment/results.json', 'w') as f:
        json.dump(results, f)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for variant in variants:
        plt.plot(results[variant]['history']['test_acc'], label=variant)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    for variant in variants:
        plt.plot(results[variant]['history']['train_loss'], label=variant)
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('residual_homotopy_experiment/comparison.png')
    plt.close()

if __name__ == "__main__":
    run_experiment()
