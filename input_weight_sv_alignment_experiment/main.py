
import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import optuna
import matplotlib.pyplot as plt
import numpy as np
import os

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# --- Data Loading ---
def get_data(data_path):
    """Fetches and prepares the mnist1d dataset."""
    try:
        # Try to load from a local file to avoid permission errors
        data = torch.load(data_path, weights_only=False)
    except FileNotFoundError:
        print("Downloading mnist1d dataset...")
        defaults = get_dataset_args()
        defaults.num_samples = 10000
        data = make_dataset(defaults)
        torch.save(data, data_path)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)
    return train_loader, test_loader

# --- Model Definition ---
class MLP(nn.Module):
    """A simple Multi-Layer Perceptron."""
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

# --- IWSVA Regularization ---
layer_inputs = {}

def get_input_hook(name):
    """Hook to capture the input of a layer."""
    def hook(model, input, output):
        layer_inputs[name] = input[0].detach()
    return hook

def iwsva_regularization(model, k=5):
    """
    Computes the Input-Weight Singular Vector Alignment (IWSVA) regularization penalty.
    The penalty is the negative Frobenius norm of the product of the top-k singular vectors,
    which encourages alignment between the input's principal components and the weight's
    principal output directions.
    """
    reg = 0.0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name in layer_inputs:
                X = layer_inputs[name]
                W = module.weight

                _, _, V_x_T = torch.linalg.svd(X, full_matrices=False)
                V_x = V_x_T.t()

                _, _, V_w_T = torch.linalg.svd(W, full_matrices=False)
                V_w = V_w_T.t()

                min_k = min(k, V_x.shape[1], V_w.shape[1])
                if min_k == 0:
                    continue

                V_x_top_k = V_x[:, :min_k]
                V_w_top_k = V_w[:, :min_k]

                alignment = torch.norm(V_x_top_k.t() @ V_w_top_k, 'fro')**2
                reg -= alignment
    return reg

# --- Training and Evaluation ---
def train_and_evaluate(model, train_loader, test_loader, lr, reg_strength=0.0, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    val_losses = []

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(get_input_hook(name)))

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            global layer_inputs
            layer_inputs = {}

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if reg_strength > 0:
                reg_loss = reg_strength * iwsva_regularization(model)
                loss += reg_loss

            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

    for hook in hooks:
        hook.remove()

    return val_losses

# --- Optuna Objective ---
def objective(trial, use_regularization, train_loader, test_loader):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    reg_strength = 0.0
    if use_regularization:
        reg_strength = trial.suggest_float("reg_strength", 1e-5, 1e-1, log=True)

    model = MLP()
    val_losses = train_and_evaluate(model, train_loader, test_loader, lr, reg_strength, epochs=5)
    return min(val_losses) if val_losses else float('inf')


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'mnist1d_data.pkl')
    plot_path = os.path.join(script_dir, 'comparison.png')

    train_loader, test_loader = get_data(data_path)

    study_baseline = optuna.create_study(direction="minimize")
    study_baseline.optimize(lambda trial: objective(trial, False, train_loader, test_loader), n_trials=15)
    best_lr_baseline = study_baseline.best_params["lr"]

    study_regularized = optuna.create_study(direction="minimize")
    study_regularized.optimize(lambda trial: objective(trial, True, train_loader, test_loader), n_trials=15)
    best_lr_regularized = study_regularized.best_params["lr"]
    best_reg_strength = study_regularized.best_params["reg_strength"]

    print(f"Best LR for Baseline: {best_lr_baseline}")
    print(f"Best LR for Regularized: {best_lr_regularized}")
    print(f"Best Reg Strength for Regularized: {best_reg_strength}")

    torch.manual_seed(0)
    np.random.seed(0)
    model_baseline = MLP()
    baseline_losses = train_and_evaluate(model_baseline, train_loader, test_loader, best_lr_baseline, epochs=25)

    torch.manual_seed(0)
    np.random.seed(0)
    model_regularized = MLP()
    regularized_losses = train_and_evaluate(model_regularized, train_loader, test_loader, best_lr_regularized, best_reg_strength, epochs=25)

    plt.figure(figsize=(10, 6))
    plt.plot(baseline_losses, label="Baseline MLP")
    plt.plot(regularized_losses, label="MLP with IWSVA Regularization")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.title("Baseline vs. IWSVA Regularization")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
