import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import plotly.graph_objects as go
import os
import argparse

from mnist1d.data import get_dataset_args, make_dataset
from light_dataloader import TensorDataLoader
from optimizer import DynamicBlendedOptimizer

# --- Model Definitions ---

class MLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, num_classes=10):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        return self.layers(x)

class GatingNet(nn.Module):
    """A small network to control the blending factor."""
    def __init__(self, input_size, hidden_size=32):
        super(GatingNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid() # Output a value between 0 and 1
        )
    def forward(self, x):
        return self.layers(x)

# --- Data Loading ---

def get_data_loaders(batch_size=128):
    # It's important to use weights_only=False for this pickle file.
    # See: https://pytorch.org/docs/stable/generated/torch.load.html
    try:
        data = torch.load('mnist1d_data.pkl', weights_only=False)
    except FileNotFoundError:
        defaults = get_dataset_args()
        data = make_dataset(defaults)
        torch.save(data, 'mnist1d_data.pkl')

    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
    X_val, y_val = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])

    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = TensorDataLoader((X_val, y_val), batch_size=batch_size)

    # Create a separate loader for meta-validation
    meta_val_loader = TensorDataLoader((X_val, y_val), batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, meta_val_loader

# --- Training and Evaluation ---

def train_epoch_meta(model, gating_net, optimizer, meta_optimizer, train_loader, meta_val_loader, criterion):
    model.train()
    gating_net.train()

    meta_val_iter = iter(meta_val_loader)

    for inputs, targets in train_loader:
        # --- Inner loop: Update model weights ---
        # We need to create the graph to allow second-order gradients for the meta-optimizer
        model.zero_grad()
        outputs = model(inputs)
        train_loss = criterion(outputs, targets)
        train_loss.backward(create_graph=True)
        optimizer.step()

        # --- Outer loop: Update gating network ---
        try:
            meta_inputs, meta_targets = next(meta_val_iter)
        except StopIteration:
            meta_val_iter = iter(meta_val_loader)
            meta_inputs, meta_targets = next(meta_val_iter)

        meta_optimizer.zero_grad()
        meta_outputs = model(meta_inputs)
        meta_loss = criterion(meta_outputs, meta_targets)
        meta_loss.backward()
        meta_optimizer.step()

def train_epoch_baseline(model, optimizer, train_loader, criterion):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# --- Optuna Objective ---

def objective(trial, optimizer_type, train_loader, val_loader, meta_val_loader):
    torch.manual_seed(42) # Ensure reproducibility for each trial
    model = MLP()
    criterion = nn.CrossEntropyLoss()

    if optimizer_type == 'DynamicBlended':
        # The gating network's input size depends on the number of model parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        gating_net = GatingNet(num_params)

        lr_adam = trial.suggest_float('lr_adam', 1e-5, 1e-2, log=True)
        lr_sgd = trial.suggest_float('lr_sgd', 1e-4, 1e-1, log=True)
        lr_gate = trial.suggest_float('lr_gate', 1e-5, 1e-2, log=True)
        momentum = trial.suggest_float('momentum', 0.8, 0.99)

        optimizer = DynamicBlendedOptimizer(model.parameters(), gating_net, lr_adam=lr_adam, lr_sgd=lr_sgd, momentum=momentum)
        meta_optimizer = optim.Adam(gating_net.parameters(), lr=lr_gate)
    elif optimizer_type == 'Adam':
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else: # SGD
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Training loop for Optuna
    n_epochs = 10 # Shorter duration for hyperparameter search
    for epoch in range(n_epochs):
        if optimizer_type == 'DynamicBlended':
            train_epoch_meta(model, gating_net, optimizer, meta_optimizer, train_loader, meta_val_loader, criterion)
        else:
            train_epoch_baseline(model, optimizer, train_loader, criterion)

    val_loss = evaluate(model, val_loader, criterion)
    return val_loss

# --- Main Execution ---

def run_final_training(optimizer_type, best_params, train_loader, val_loader, meta_val_loader, n_epochs=25):
    torch.manual_seed(42)
    model = MLP()
    criterion = nn.CrossEntropyLoss()

    if optimizer_type == 'DynamicBlended':
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        gating_net = GatingNet(num_params)
        optimizer = DynamicBlendedOptimizer(model.parameters(), gating_net, **best_params['optimizer'])
        meta_optimizer = optim.Adam(gating_net.parameters(), lr=best_params['lr_gate'])
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), **best_params)
    else: # SGD
        optimizer = optim.SGD(model.parameters(), **best_params)

    history = []
    for epoch in range(n_epochs):
        if optimizer_type == 'DynamicBlended':
            train_epoch_meta(model, gating_net, optimizer, meta_optimizer, train_loader, meta_val_loader, criterion)
        else:
            train_epoch_baseline(model, optimizer, train_loader, criterion)

        val_loss = evaluate(model, val_loader, criterion)
        history.append(val_loss)
        print(f'Epoch {epoch+1}/{n_epochs}, {optimizer_type} Val Loss: {val_loss:.4f}')

    return history

def main():
    parser = argparse.ArgumentParser(description='Run dynamic optimizer blending experiment.')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials.')
    parser.add_argument('--n_epochs', type=int, default=25, help='Number of epochs for final training.')
    args = parser.parse_args()

    # Get data
    train_loader, val_loader, meta_val_loader = get_data_loaders()

    # Run Optuna studies
    optimizer_types = ['DynamicBlended', 'Adam', 'SGD']
    best_params = {}

    for opt_type in optimizer_types:
        print(f"--- Running Optuna for {opt_type} ---")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, opt_type, train_loader, val_loader, meta_val_loader), n_trials=args.n_trials)

        # Store params in a format suitable for optimizer instantiation
        if opt_type == 'DynamicBlended':
            best_params[opt_type] = {
                'optimizer': {
                    'lr_adam': study.best_params['lr_adam'],
                    'lr_sgd': study.best_params['lr_sgd'],
                    'momentum': study.best_params['momentum']
                },
                'lr_gate': study.best_params['lr_gate']
            }
        else:
            best_params[opt_type] = study.best_params

        print(f"Best params for {opt_type}: {study.best_params}")

        # Save Optuna plots
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(f"{opt_type}_optimization_history.png")
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(f"{opt_type}_param_importances.png")

    # Run final training with best params and collect history
    histories = {}
    for opt_type in optimizer_types:
        print(f"--- Running Final Training for {opt_type} ---")
        histories[opt_type] = run_final_training(opt_type, best_params[opt_type], train_loader, val_loader, meta_val_loader, n_epochs=args.n_epochs)

    # Plot final comparison
    fig = go.Figure()
    for opt_type, history in histories.items():
        fig.add_trace(go.Scatter(y=history, mode='lines', name=opt_type))

    fig.update_layout(
        title='Final Comparison of Optimizer Validation Loss',
        xaxis_title='Epoch',
        yaxis_title='Validation Loss',
        legend_title='Optimizer'
    )
    fig.write_image("final_comparison.png")
    print("Experiment complete. Plots saved.")

if __name__ == '__main__':
    # Fix for Optuna/Plotly in headless environments
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"
    main()
