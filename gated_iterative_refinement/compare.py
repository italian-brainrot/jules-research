import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args

# Add current directory to sys.path to import model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import GIRModel, MLPBaseline, GRUBaseline

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model, train_loader, val_loader, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    num_epochs = config['epochs']

    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits_list, _ = model(batch_x)

            # Loss is sum of CE losses at each step
            loss = 0
            for logits in logits_list:
                loss += F.cross_entropy(logits, batch_y)

            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    accuracies = []
    with torch.no_grad():
        # Get accuracy for each step
        X_test, y_test = val_loader
        X_test, y_test = X_test.to(device), y_test.to(device)
        logits_list, _ = model(X_test)

        for logits in logits_list:
            preds = logits.argmax(dim=-1)
            acc = (preds == y_test).float().mean().item()
            accuracies.append(acc)

    return accuracies

def objective(trial, model_name, device, data):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = 128
    num_steps = 5
    epochs = 20
    batch_size = 128

    X_train, y_train, X_test, y_test = data
    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    if model_name == 'mlp':
        model = MLPBaseline(40, hidden_dim, 10, num_layers=4).to(device)
    elif model_name == 'gru':
        model = GRUBaseline(40, hidden_dim, 10, num_steps=num_steps).to(device)
    elif model_name.startswith('gir_'):
        strategy = model_name.split('_', 1)[1]
        model = GIRModel(40, hidden_dim, 10, num_steps=num_steps, strategy=strategy).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    config = {'lr': lr, 'epochs': epochs}
    accuracies = train_model(model, train_loader, (X_test, y_test), config, device)

    return accuracies[-1] # Return final step accuracy

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_to_test = ['mlp', 'gru', 'gir_none', 'gir_max_prob', 'gir_entropy', 'gir_learned']
    results = {}

    X_train_full, y_train_full, X_test_full, y_test_full = get_data()
    data = (X_train_full, y_train_full, X_test_full, y_test_full)

    for name in models_to_test:
        print(f"Testing model: {name}")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, name, device, data), n_trials=10)

        best_lr = study.best_params['lr']
        print(f"Best LR for {name}: {best_lr}")

        # Train final model with best LR
        train_loader = TensorDataLoader((X_train_full, y_train_full), batch_size=128, shuffle=True)

        hidden_dim = 128
        num_steps = 5
        if name == 'mlp':
            model = MLPBaseline(40, hidden_dim, 10, num_layers=4).to(device)
        elif name == 'gru':
            model = GRUBaseline(40, hidden_dim, 10, num_steps=num_steps).to(device)
        else:
            strategy = name.split('_', 1)[1]
            model = GIRModel(40, hidden_dim, 10, num_steps=num_steps, strategy=strategy).to(device)

        config = {'lr': best_lr, 'epochs': 40} # More epochs for final
        accuracies = train_model(model, train_loader, (X_test_full, y_test_full), config, device)

        results[name] = {
            'best_lr': best_lr,
            'accuracies': accuracies,
            'final_accuracy': accuracies[-1]
        }

    with open('gated_iterative_refinement/results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Plotting
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        accs = data['accuracies']
        if len(accs) > 1:
            plt.plot(range(len(accs)), accs, label=name, marker='o')
        else:
            plt.axhline(y=accs[0], label=f"{name} (baseline)", linestyle='--')

    plt.xlabel('Iteration Step')
    plt.ylabel('Test Accuracy')
    plt.title('Iterative Refinement Accuracy over Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('gated_iterative_refinement/accuracy_plot.png')
    plt.close()

if __name__ == "__main__":
    run_experiment()
