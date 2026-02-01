import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
import numpy as np
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data():
    if hasattr(get_data, "cache"):
        return get_data.cache
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X = torch.tensor(data['x'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    # Split X, y into train and val
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    get_data.cache = (X_train, y_train, X_val, y_val, X_test, y_test)
    return get_data.cache

class MLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def logit_cohesion_loss(logits, targets):
    unique_targets = torch.unique(targets)
    loss = 0.0
    count = 0
    for t in unique_targets:
        mask = (targets == t)
        if mask.sum() > 1:
            class_logits = logits[mask]
            mean_logits = class_logits.mean(dim=0)
            # Squared Euclidean distance from mean
            loss += torch.mean(torch.sum((class_logits - mean_logits)**2, dim=1))
            count += 1
    if count > 0:
        return loss / count
    return torch.tensor(0.0, device=logits.device)

def train_eval(lr, epochs, method="ce", method_params=None, is_final=False):
    X_train, y_train, X_val, y_val, X_test, y_test = get_data()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    if method == "ls":
        criterion = nn.CrossEntropyLoss(label_smoothing=method_params.get("epsilon", 0.1))

    for epoch in range(epochs):
        model.train()
        for inputs, targets in dl_train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)

            loss = criterion(logits, targets)

            if method == "lcl":
                lcl_weight = method_params.get("lambda", 0.1)
                lcl_loss = logit_cohesion_loss(logits, targets)
                loss += lcl_weight * lcl_loss

            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        if is_final:
            eval_X, eval_y = X_test, y_test
        else:
            eval_X, eval_y = X_val, y_val

        test_logits = model(eval_X.to(device))
        preds = test_logits.argmax(dim=1)
        acc = (preds == eval_y.to(device)).float().mean().item()

    return acc

def objective(trial, method):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    params = {}
    if method == "ls":
        params["epsilon"] = trial.suggest_float("epsilon", 0.01, 0.3)
    elif method == "lcl":
        params["lambda"] = trial.suggest_float("lambda", 1e-4, 1.0, log=True)

    # Tuning epochs
    epochs = 40
    return train_eval(lr, epochs, method, params, is_final=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    n_trials = 30
    epochs = 100
    if args.smoke_test:
        n_trials = 1
        epochs = 1

    results = {}

    for method in ["ce", "ls", "lcl"]:
        print(f"Tuning {method}...")
        study = optuna.create_study(direction="maximize")
        if args.smoke_test:
            study.optimize(lambda t: train_eval(1e-3, 1, method, {"epsilon": 0.1, "lambda": 0.1}, is_final=False), n_trials=n_trials)
        else:
            study.optimize(lambda t: objective(t, method), n_trials=n_trials)

        best_params = study.best_params
        if args.smoke_test:
            best_params = {"lr": 1e-3}
            if method == "ls": best_params["epsilon"] = 0.1
            if method == "lcl": best_params["lambda"] = 0.1

        print(f"Training final {method} with best params: {best_params}")
        final_acc = train_eval(best_params["lr"], epochs, method, best_params, is_final=True)
        results[method] = {"best_params": best_params, "final_acc": final_acc}

    with open("logit_cohesion_experiment/results.txt", "w") as f:
        for method, res in results.items():
            f.write(f"Method: {method}\n")
            f.write(f"Best Params: {res['best_params']}\n")
            f.write(f"Final Test Accuracy: {res['final_acc']:.4f}\n")
            f.write("-" * 20 + "\n")

    print("\nFinal Results:")
    for method, res in results.items():
        print(f"{method}: {res['final_acc']:.4f} with {res['best_params']}")
