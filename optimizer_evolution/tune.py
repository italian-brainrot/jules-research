import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import mnist1d.data as mnist1d_data
from optimizer import EvolvedOptimizer
from evolve import SimpleNN
from gp import func_map

# Define the evolved optimizer expression here
# This should be the output from the evolution process
# For now, let's use a placeholder expression
# Example: evolved_expr = App(Var('mul'), App(Var('neg'), App(Var('lr'), Var('g')))) # Simple SGD
evolved_expr_str = "sqrt(mul(m, add(m, mul(sqrt(v), neg(one)))))"

def objective(trial, optimizer_name, train_loader):
    model = SimpleNN()
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    if optimizer_name == "evolved":
        try:
            optimizer = EvolvedOptimizer(model.parameters(), evolved_expr_str, lr=lr)
        except Exception as e:
            print(f"Error creating optimizer: {e}")
            return float('inf')
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Unknown optimizer")

    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx >= 20:
            break

    return total_loss / 20

def main():
    # Load mnist1d data
    args = mnist1d_data.get_dataset_args()
    data = mnist1d_data.get_dataset(args)
    X_train, y_train = torch.from_numpy(data['x']).float(), torch.from_numpy(data['y']).long()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Tune Evolved Optimizer
    study_evolved = optuna.create_study(direction="minimize")
    study_evolved.optimize(lambda trial: objective(trial, "evolved", train_loader), n_trials=20)
    print("Best hyperparameters for Evolved Optimizer: ", study_evolved.best_trial.params)

    # Tune Adam Optimizer
    study_adam = optuna.create_study(direction="minimize")
    study_adam.optimize(lambda trial: objective(trial, "adam", train_loader), n_trials=20)
    print("Best hyperparameters for Adam Optimizer: ", study_adam.best_trial.params)

if __name__ == "__main__":
    main()
