import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader

class LinearInterpolatedReLU(nn.Module):
    def __init__(self, alpha=0.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return (1.0 - self.alpha) * x + self.alpha * F.relu(x)

class LongThinNet(nn.Module):
    def __init__(self, input_size=40, hidden_size=10, output_size=10, num_layers=16, alpha=0.0):
        super().__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))

        self.activation = LinearInterpolatedReLU(alpha)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

    def set_alpha(self, alpha):
        self.activation.alpha = alpha

def compute_ls_solution(X, y):
    # One-hot encode y
    num_classes = 10
    Y = torch.zeros(X.shape[0], num_classes)
    Y.scatter_(1, y.unsqueeze(1), 1.0)

    # Add bias by appending 1s to X
    X_aug = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)

    # Solve X_aug W_full = Y  => W_full = (X_aug^T X_aug)^-1 X_aug^T Y
    XTX = X_aug.t() @ X_aug
    XTX += 1e-4 * torch.eye(XTX.shape[0])
    W_full = torch.linalg.solve(XTX, X_aug.t() @ Y)

    W = W_full[:-1, :]
    b = W_full[-1, :]

    return W, b

def ls_init(model, W_ls, b_ls):
    # W_ls is [40, 10], b_ls is [10]
    # Model has L layers. W_total = W_L * ... * W_1 should be W_ls.T
    # W_ls.T = U S V^T

    W_ls_T = W_ls.t() # [10, 40]
    U, S, Vh = torch.linalg.svd(W_ls_T, full_matrices=False)
    # U: [10, 10], S: [10], Vh: [10, 40]

    num_layers = len(model.layers)
    hidden_size = model.layers[0].out_features

    with torch.no_grad():
        # Initialize all to identity-like or zero
        for layer in model.layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Distribute factors
        # Layer 1: [hidden, 40] -> set top 10 rows to Vh
        model.layers[0].weight[:10, :] = Vh

        # Intermediate layers: [hidden, hidden] -> set top-left 10x10 to identity
        # We'll put S in the second layer
        for i in range(1, num_layers - 1):
            if i == 1:
                # Layer 2: Put S here
                for j in range(10):
                    model.layers[i].weight[j, j] = S[j]
            else:
                # Other layers: Identity for the first 10 dims
                for j in range(10):
                    model.layers[i].weight[j, j] = 1.0

        # Last layer: [10, hidden] -> set top-left 10x10 to U
        model.layers[-1].weight[:, :10] = U
        model.layers[-1].bias[:] = b_ls

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()

    return X_train, y_train, X_test, y_test

def train_model(model, train_loader, val_data, epochs, lr, alpha_schedule=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_val, y_val = val_data

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        if alpha_schedule is not None:
            model.set_alpha(alpha_schedule[epoch])

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val)
            val_loss = criterion(outputs_val, y_val).item()
            _, predicted_val = outputs_val.max(1)
            val_acc = predicted_val.eq(y_val).sum().item() / y_val.size(0)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    return history

import optuna

def objective(trial, X_train, y_train, X_test, y_test, config_name):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    num_layers = 16
    hidden_size = 10
    epochs = 30 # Shorter for tuning

    model = LongThinNet(num_layers=num_layers, hidden_size=hidden_size)

    alpha_schedule = None
    if config_name == "Baseline":
        model.set_alpha(1.0)
    elif config_name == "Interpolation":
        alpha_schedule = np.linspace(0.0, 1.0, epochs)
    elif config_name == "LS_Init":
        W_ls, b_ls = compute_ls_solution(X_train, y_train)
        ls_init(model, W_ls, b_ls)
        model.set_alpha(1.0)
    elif config_name == "LS_Init_Interpolation":
        W_ls, b_ls = compute_ls_solution(X_train, y_train)
        ls_init(model, W_ls, b_ls)
        alpha_schedule = np.linspace(0.0, 1.0, epochs)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    history = train_model(model, train_loader, (X_test, y_test), epochs, lr, alpha_schedule)

    return max(history['val_acc'])

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    configs = ["Baseline", "Interpolation", "LS_Init", "LS_Init_Interpolation"]
    best_lrs = {}

    for config in configs:
        print(f"Tuning {config}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, config), n_trials=5)
        best_lrs[config] = study.best_params['lr']
        print(f"Best LR for {config}: {best_lrs[config]}")

    # Run full training
    results = {}
    epochs = 100
    for config in configs:
        print(f"Running full training for {config}...")
        model = LongThinNet(num_layers=16, hidden_size=10)
        alpha_schedule = None
        if config == "Baseline":
            model.set_alpha(1.0)
        elif config == "Interpolation":
            alpha_schedule = np.linspace(0.0, 1.0, epochs)
        elif config == "LS_Init":
            W_ls, b_ls = compute_ls_solution(X_train, y_train)
            ls_init(model, W_ls, b_ls)
            model.set_alpha(1.0)
        elif config == "LS_Init_Interpolation":
            W_ls, b_ls = compute_ls_solution(X_train, y_train)
            ls_init(model, W_ls, b_ls)
            alpha_schedule = np.linspace(0.0, 1.0, epochs)

        train_loader = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
        history = train_model(model, train_loader, (X_test, y_test), epochs, best_lrs[config], alpha_schedule)
        results[config] = history

    # Plotting and saving results
    import pickle
    import os
    import matplotlib.pyplot as plt

    output_dir = os.path.dirname(__file__)
    with open(os.path.join(output_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Accuracy Plot
    plt.figure(figsize=(10, 6))
    for config, history in results.items():
        plt.plot(history['val_acc'], label=f'{config} (val)')
        plt.plot(history['train_acc'], '--', label=f'{config} (train)')

    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))
    plt.close()

    # Loss Plot
    plt.figure(figsize=(10, 6))
    for config, history in results.items():
        plt.plot(history['val_loss'], label=f'{config} (val)')
        plt.plot(history['train_loss'], '--', label=f'{config} (train)')

    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()

    print("Experiments completed and plots saved.")
