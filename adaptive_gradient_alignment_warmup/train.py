import torch
import torch.nn as nn
import optuna
import numpy as np
import matplotlib.pyplot as plt
import os
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from adaptive_gradient_alignment_warmup.optimizer import AGAWOptimizer

class MLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def get_data():
    args = get_dataset_args()
    args.num_samples = 5000
    data = make_dataset(args)
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

def train_model(mode, config, data, device, num_epochs=50):
    X_train, y_train, X_val, y_val, X_test, y_test = data
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

    model = MLP().to(device)
    target_lr = config['lr']

    # Use a fresh optimizer for each call
    base_optimizer = torch.optim.Adam(model.parameters(), lr=target_lr)

    optimizer = base_optimizer
    scheduler = None

    if mode == 'Warmup':
        warmup_steps = config['warmup_steps']
        # Linear warmup using LambdaLR
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(base_optimizer, lr_lambda)
    elif mode == 'AGAW':
        optimizer = AGAWOptimizer(
            base_optimizer,
            target_lr=target_lr,
            gamma=config['gamma'],
            warmup_steps_nominal=config['warmup_steps_nominal']
        )

    history = {'train_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            # Record current LR
            if mode == 'AGAW':
                history['lr'].append(optimizer.current_lr)
            else:
                history['lr'].append(base_optimizer.param_groups[0]['lr'])

            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = model(X_val.to(device))
            v_acc = (v_logits.argmax(dim=1) == y_val.to(device)).float().mean().item()

        history['train_loss'].append(epoch_loss / len(dl_train))
        history['val_acc'].append(v_acc)

    # Test accuracy
    model.eval()
    with torch.no_grad():
        t_logits = model(X_test.to(device))
        t_acc = (t_logits.argmax(dim=1) == y_test.to(device)).float().mean().item()

    return t_acc, history

def objective(trial, mode, data, device):
    config = {
        'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    }
    if mode == 'Warmup':
        config['warmup_steps'] = trial.suggest_int('warmup_steps', 50, 500)
    elif mode == 'AGAW':
        config['gamma'] = trial.suggest_float('gamma', 0.5, 4.0)
        config['warmup_steps_nominal'] = trial.suggest_int('warmup_steps_nominal', 50, 500)

    acc, _ = train_model(mode, config, data, device, num_epochs=15)
    return acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = get_data()

    modes = ['Baseline', 'Warmup', 'AGAW']
    best_configs = {}
    final_histories = {}
    final_results = {}

    for mode in modes:
        print(f"--- Tuning {mode} ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, mode, data, device), n_trials=15)
        best_configs[mode] = study.best_params
        print(f"Best config for {mode}: {best_configs[mode]}")

        print(f"--- Final Training {mode} ---")
        acc, history = train_model(mode, best_configs[mode], data, device, num_epochs=50)
        final_results[mode] = acc
        final_histories[mode] = history
        print(f"Final Test Accuracy for {mode}: {acc:.4f}")

    # Plotting
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    for mode in modes:
        plt.plot(final_histories[mode]['val_acc'], label=mode)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 3, 2)
    for mode in modes:
        plt.plot(final_histories[mode]['lr'], label=mode)
    plt.title('Learning Rate')
    plt.xlabel('Step')
    plt.legend()

    plt.subplot(1, 3, 3)
    for mode in modes:
        plt.plot(final_histories[mode]['train_loss'], label=mode)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('adaptive_gradient_alignment_warmup/comparison.png')

    with open('adaptive_gradient_alignment_warmup/results.txt', 'w') as f:
        for mode in modes:
            f.write(f"Mode: {mode}\n")
            f.write(f"  Best Config: {best_configs[mode]}\n")
            f.write(f"  Test Accuracy: {final_results[mode]:.4f}\n\n")

if __name__ == "__main__":
    main()
