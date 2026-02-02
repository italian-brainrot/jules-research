import optuna
import torch
from train import get_data, train_model
from light_dataloader import TensorDataLoader
from model import SoftWalshNetwork, MLP
import json
import os

# Cache data to avoid repeated dataset creation
X_train, y_train, X_test, y_test = get_data()

def objective_mlp(trial):
    train_loader = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=256, shuffle=False)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    hidden_size = trial.suggest_int("hidden_size", 64, 512)
    num_layers = trial.suggest_int("num_layers", 2, 4)

    model = MLP(40, hidden_size, 10, num_layers=num_layers)
    acc = train_model(model, train_loader, test_loader, epochs=15, lr=lr, weight_decay=weight_decay)
    return acc

def objective_swn(trial):
    train_loader = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=256, shuffle=False)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    num_terms = trial.suggest_int("num_terms", 64, 512)
    sparsity_lambda = trial.suggest_float("sparsity_lambda", 1e-6, 1e-2, log=True)
    deep = trial.suggest_categorical("deep", [True, False])
    init_scale = trial.suggest_float("init_scale", 1e-3, 0.5, log=True)

    model = SoftWalshNetwork(40, num_terms, 10, deep=deep, init_scale=init_scale)
    acc = train_model(model, train_loader, test_loader, epochs=15, lr=lr, weight_decay=weight_decay, sparsity_lambda=sparsity_lambda)
    return acc

if __name__ == "__main__":
    study_mlp = optuna.create_study(direction="maximize")
    print("Tuning MLP...")
    study_mlp.optimize(objective_mlp, n_trials=8)

    study_swn = optuna.create_study(direction="maximize")
    print("Tuning SWN...")
    study_swn.optimize(objective_swn, n_trials=8)

    results = {
        "mlp": study_mlp.best_params,
        "swn": study_swn.best_params,
        "mlp_acc": study_mlp.best_value,
        "swn_acc": study_swn.best_value
    }
    with open("soft_walsh_experiment/best_hparams.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Best MLP:", study_mlp.best_params)
    print("Best SWN:", study_swn.best_params)
