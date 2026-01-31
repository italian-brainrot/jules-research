import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add current directory to path so we can import model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import MLP, compute_jacobian_penalties

from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 8000  # Sufficient samples for regularization to matter
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data['x_test']).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(config, X_train, y_train, X_test, y_test, num_epochs=20, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)

    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            if config.get('use_jfnr') or config.get('use_cjor'):
                # We only compute penalties if needed
                jfnr, cjor = compute_jacobian_penalties(
                    model, batch_x,
                    compute_cjor=config.get('use_cjor', False),
                    compute_jfnr=config.get('use_jfnr', False)
                )
                if config.get('use_jfnr'):
                    loss += config['lambda_jfnr'] * jfnr
                if config.get('use_cjor'):
                    loss += config['lambda_cjor'] * cjor

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_acc = (test_outputs.argmax(dim=1) == y_test).float().mean().item()

        history['train_loss'].append(total_loss / len(train_loader))
        history['test_acc'].append(test_acc)

        if verbose and (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Test Acc: {test_acc:.4f}")

    return history, model

def objective(trial, config_template, X_train, y_train, X_test, y_test):
    config = config_template.copy()
    config['lr'] = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    if config.get('use_jfnr'):
        config['lambda_jfnr'] = trial.suggest_float('lambda_jfnr', 1e-5, 1e-1, log=True)
    if config.get('use_cjor'):
        config['lambda_cjor'] = trial.suggest_float('lambda_cjor', 1e-4, 1.0, log=True)

    # Use fewer epochs for tuning to save time
    history, _ = train_model(config, X_train, y_train, X_test, y_test, num_epochs=15)
    # Return best accuracy achieved during training
    return max(history['test_acc'])

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    configs = [
        {'name': 'Baseline', 'use_jfnr': False, 'use_cjor': False},
        {'name': 'JFNR', 'use_jfnr': True, 'use_cjor': False},
        {'name': 'CJOR', 'use_jfnr': False, 'use_cjor': True},
        {'name': 'CJOR+JFNR', 'use_jfnr': True, 'use_cjor': True},
    ]

    results = {}
    best_configs = {}

    experiment_dir = 'jacobian_orthogonality_experiment'

    for conf in configs:
        print(f"\n--- Tuning {conf['name']} ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, conf, X_train, y_train, X_test, y_test), n_trials=20)

        best_params = study.best_params
        full_config = conf.copy()
        full_config.update(best_params)
        best_configs[conf['name']] = full_config

        print(f"Best params for {conf['name']}: {best_params}")

        print(f"Final training for {conf['name']}...")
        history, _ = train_model(full_config, X_train, y_train, X_test, y_test, num_epochs=50, verbose=True)
        results[conf['name']] = history

    # Plotting
    plt.figure(figsize=(10, 6))
    for name, history in results.items():
        plt.plot(history['test_acc'], label=name)
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(experiment_dir, 'accuracy_comparison.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, history in results.items():
        plt.plot(history['train_loss'], label=name)
    plt.title('Train Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(experiment_dir, 'loss_comparison.png'))
    plt.close()

    # Save best configs and final results
    with open(os.path.join(experiment_dir, 'results.txt'), 'w') as f:
        for name, history in results.items():
            f.write(f"--- {name} ---\n")
            f.write(f"Best Test Acc: {max(history['test_acc']):.4f}\n")
            f.write(f"Final Test Acc: {history['test_acc'][-1]:.4f}\n")
            f.write(f"Config: {best_configs[name]}\n\n")

if __name__ == "__main__":
    run_experiment()
