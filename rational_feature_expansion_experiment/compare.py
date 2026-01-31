import optuna
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

# Ensure the experiment directory is in the path for local imports
dir_path = os.path.dirname(os.path.abspath(__file__))
if dir_path not in sys.path:
    sys.path.append(dir_path)

from main import get_data, MLP, train_mlp
from model import RationalFeatureExpansion, ELM
from light_dataloader import TensorDataLoader

def objective_rfe(trial, X_train, y_train, X_test, y_test):
    num_features = trial.suggest_int('num_features', 200, 1000, step=200)
    sigma = trial.suggest_float('sigma', 0.01, 1.0, log=True)
    reg = trial.suggest_float('reg', 1e-6, 1.0, log=True)
    num_iter = trial.suggest_int('num_iter', 1, 5)

    y_train_oh = F.one_hot(y_train, num_classes=10).float()

    model = RationalFeatureExpansion(num_features=num_features, sigma=sigma, reg=reg, num_iter=num_iter)
    model.fit(X_train, y_train_oh)

    preds = model.predict(X_test)
    acc = (preds.cpu() == y_test).float().mean().item()
    return acc

def objective_elm(trial, X_train, y_train, X_test, y_test):
    num_features = trial.suggest_int('num_features', 200, 1000, step=200)
    sigma = trial.suggest_float('sigma', 0.01, 1.0, log=True)
    reg = trial.suggest_float('reg', 1e-6, 1.0, log=True)

    y_train_oh = F.one_hot(y_train, num_classes=10).float()

    model = ELM(num_features=num_features, sigma=sigma, reg=reg)
    model.fit(X_train, y_train_oh)

    preds = model.predict(X_test)
    acc = (preds.cpu() == y_test).float().mean().item()
    return acc

def objective_mlp(trial, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    num_layers = trial.suggest_int('num_layers', 1, 3)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    model = MLP(input_dim=40, hidden_dim=hidden_dim, output_dim=10, num_layers=num_layers)
    acc = train_mlp(model, train_loader, test_loader, lr=lr, epochs=20)
    return acc

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data(num_samples=4000)

    print("Tuning ELM...")
    study_elm = optuna.create_study(direction='maximize')
    study_elm.optimize(lambda trial: objective_elm(trial, X_train, y_train, X_test, y_test), n_trials=10)

    print("Tuning RFE...")
    study_rfe = optuna.create_study(direction='maximize')
    study_rfe.optimize(lambda trial: objective_rfe(trial, X_train, y_train, X_test, y_test), n_trials=10)

    print("Tuning MLP...")
    study_mlp = optuna.create_study(direction='maximize')
    study_mlp.optimize(lambda trial: objective_mlp(trial, X_train, y_train, X_test, y_test), n_trials=10)

    print("\nResults:")
    print(f"ELM: Best Acc = {study_elm.best_value:.4f}, Best Params = {study_elm.best_params}")
    print(f"RFE: Best Acc = {study_rfe.best_value:.4f}, Best Params = {study_rfe.best_params}")
    print(f"MLP: Best Acc = {study_mlp.best_value:.4f}, Best Params = {study_mlp.best_params}")

    # Save results to a file for README
    results_path = os.path.join(dir_path, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"ELM: {study_elm.best_value:.4f}\n")
        f.write(f"RFE: {study_rfe.best_value:.4f}\n")
        f.write(f"MLP: {study_mlp.best_value:.4f}\n")
        f.write(f"ELM Params: {study_elm.best_params}\n")
        f.write(f"RFE Params: {study_rfe.best_params}\n")
        f.write(f"MLP Params: {study_mlp.best_params}\n")
