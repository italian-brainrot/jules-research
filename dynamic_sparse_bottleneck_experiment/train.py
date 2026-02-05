import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import numpy as np
import optuna
from model import BaselineMLP, DSBMLP, FSBMLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model_class, lr, epochs=50, batch_size=128, device='cpu', data=None, save_path=None):
    if data is None:
        X_train, y_train, X_test, y_test = get_data()
    else:
        X_train, y_train, X_test, y_test = data
    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    model = model_class().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_model_state = None
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = model(X_test.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total = y_test.size(0)
            correct = (predicted == y_test.to(device)).sum().item()

        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
            if save_path:
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if save_path and best_model_state:
        torch.save(best_model_state, save_path)

    return best_acc

def objective(trial, model_name, data=None):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_name == "baseline":
        model_class = BaselineMLP
    elif model_name == "dsb":
        model_class = DSBMLP
    elif model_name == "fsb":
        model_class = FSBMLP
    else:
        raise ValueError("Unknown model name")

    return train_model(model_class, lr, epochs=30, data=data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    data = get_data()
    if args.tune:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, args.model, data=data), n_trials=20)
        print(f"Best trial for {args.model}: {study.best_trial.params}")
    else:
        acc = train_model(eval(f"{args.model.upper()}MLP") if args.model != "baseline" else BaselineMLP, args.lr, data=data)
        print(f"Accuracy for {args.model}: {acc}")
