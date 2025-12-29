import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import numpy as np
import matplotlib.pyplot as plt
import copy
import optuna
import os

from pruning import compute_taylor_saliency, prune_model_with_saliency, magnitude_prune

# --- Model Definition ---
class MLP(nn.Module):
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

# --- Data Loading ---
def get_dataloaders(batch_size=128):
    defaults = get_dataset_args()
    defaults.num_samples = 8000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])

    # Use a subset for saliency calculation to speed it up
    saliency_loader = TensorDataLoader((X_train[:1000], y_train[:1000]), batch_size=batch_size)
    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=batch_size)
    return train_loader, test_loader, saliency_loader

# --- Training and Evaluation ---
def train(model, train_loader, optimizer, loss_fn, epochs=1):
    model.train()
    # Store the original weights to apply the mask after update
    original_weights = {name: p.data.clone() for name, p in model.named_parameters()}
    mask = {name: (p.data != 0).float() for name, p in model.named_parameters()}

    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            # Re-apply the pruning mask after optimizer step
            with torch.no_grad():
                for name, p in model.named_parameters():
                    p.data.mul_(mask[name])

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

def get_model_sparsity(model):
    """Calculates the sparsity of the entire model."""
    total_params = 0
    zero_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            total_params += p.numel()
            zero_params += (p.data == 0).sum().item()
    if total_params == 0:
        return 0.0
    return zero_params / total_params

# --- Pruning Experiment ---
def run_pruning_experiment(lr, pruning_method, train_loader, test_loader, saliency_loader):
    torch.manual_seed(42)
    model = MLP()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Initial training
    train(model, train_loader, optimizer, loss_fn, epochs=5)

    accuracies = []
    sparsity_levels = np.linspace(0.0, 0.95, 20)

    # Evaluate initial model
    initial_acc = evaluate(model, test_loader)
    accuracies.append(initial_acc)

    for i, sparsity in enumerate(sparsity_levels[1:]): # Skip 0%
        # Determine the prune ratio for this step
        # This is the ratio of remaining weights to prune
        current_sparsity = get_model_sparsity(model)
        prune_ratio = (sparsity - current_sparsity) / (1 - current_sparsity) if (1 - current_sparsity) > 0 else 1.0
        prune_ratio = max(0, min(1, prune_ratio))


        if pruning_method == 'taylor':
            saliency_scores = compute_taylor_saliency(model, loss_fn, saliency_loader)
            prune_model_with_saliency(model, saliency_scores, prune_ratio)
        elif pruning_method == 'magnitude':
            magnitude_prune(model, prune_ratio)
        else:
            raise ValueError("Unknown pruning method")

        # Retrain after pruning
        optimizer = optim.Adam(model.parameters(), lr=lr) # Re-init optimizer
        train(model, train_loader, optimizer, loss_fn, epochs=2)

        acc = evaluate(model, test_loader)
        accuracies.append(acc)

    return accuracies

# --- Optuna Objective ---
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    pruning_method = trial.study.user_attrs['pruning_method']

    # Using cached dataloaders
    train_loader = trial.study.user_attrs['train_loader']
    test_loader = trial.study.user_attrs['test_loader']
    saliency_loader = trial.study.user_attrs['saliency_loader']

    try:
        accuracies = run_pruning_experiment(lr, pruning_method, train_loader, test_loader, saliency_loader)
        # We want to maximize the area under the accuracy-sparsity curve
        return np.mean(accuracies)
    except Exception as e:
        # Handle potential errors during training (e.g., CUDA out of memory if GPU was used)
        print(f"Trial failed with error: {e}")
        return -1.0


# --- Main Execution ---
if __name__ == '__main__':
    train_loader, test_loader, saliency_loader = get_dataloaders()
    sparsity_levels = np.linspace(0.0, 0.95, 20)

    # --- Find best LR for Taylor Pruning ---
    study_taylor = optuna.create_study(direction='maximize')
    study_taylor.set_user_attr('pruning_method', 'taylor')
    study_taylor.set_user_attr('train_loader', train_loader)
    study_taylor.set_user_attr('test_loader', test_loader)
    study_taylor.set_user_attr('saliency_loader', saliency_loader)
    study_taylor.optimize(objective, n_trials=15)
    best_lr_taylor = study_taylor.best_params['lr']
    print(f"Best LR for Taylor Pruning: {best_lr_taylor}")

    # --- Find best LR for Magnitude Pruning ---
    study_magnitude = optuna.create_study(direction='maximize')
    study_magnitude.set_user_attr('pruning_method', 'magnitude')
    study_magnitude.set_user_attr('train_loader', train_loader)
    study_magnitude.set_user_attr('test_loader', test_loader)
    study_magnitude.set_user_attr('saliency_loader', saliency_loader)
    study_magnitude.optimize(objective, n_trials=15)
    best_lr_magnitude = study_magnitude.best_params['lr']
    print(f"Best LR for Magnitude Pruning: {best_lr_magnitude}")

    # --- Run final experiments with best LRs ---
    print("Running final experiment for Taylor Pruning...")
    taylor_accuracies = run_pruning_experiment(best_lr_taylor, 'taylor', train_loader, test_loader, saliency_loader)

    print("Running final experiment for Magnitude Pruning...")
    magnitude_accuracies = run_pruning_experiment(best_lr_magnitude, 'magnitude', train_loader, test_loader, saliency_loader)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(sparsity_levels, taylor_accuracies, marker='o', linestyle='-', label='Taylor Pruning')
    plt.plot(sparsity_levels, magnitude_accuracies, marker='x', linestyle='--', label='Magnitude Pruning')
    plt.title('Model Accuracy vs. Sparsity Level')
    plt.xlabel('Sparsity Level')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.legend()
    plt.ylim(0.0, 1.0)

    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'pruning_comparison.png'))
    print("Plot saved to pruning_comparison.png")
    plt.show()
