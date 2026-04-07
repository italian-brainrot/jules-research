import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import numpy as np
from model import BaselineMLP, TropicalMLP, TropicalAugmentedMLP
import argparse
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model, dl_train, dl_test, epochs=20, lr=1e-3, weight_decay=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for x, y in dl_train:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dl_test:
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
    return best_acc

def objective(trial, model_type):
    X_train, y_train, X_test, y_test = get_data()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 4)

    if model_type == "baseline":
        model = BaselineMLP(40, hidden_dim, 10, num_layers=num_layers)
    elif model_type == "tropical":
        init_beta = trial.suggest_float("init_beta", 0.1, 10.0, log=True)
        model = TropicalMLP(40, hidden_dim, 10, num_layers=num_layers, init_beta=init_beta)
    elif model_type == "augmented":
        init_beta = trial.suggest_float("init_beta", 0.1, 10.0, log=True)
        model = TropicalAugmentedMLP(40, hidden_dim, 10, num_layers=num_layers, init_beta=init_beta)

    return train_model(model, dl_train, dl_test, lr=lr, weight_decay=weight_decay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["baseline", "tropical", "augmented"], required=True)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args.model), n_trials=args.trials)

    print(f"Best trial for {args.model}:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")

    # Save results to a file
    with open(f"results_{args.model}.txt", "w") as f:
        f.write(f"Best Value: {study.best_value}\n")
        f.write(f"Best Params: {study.best_params}\n")
