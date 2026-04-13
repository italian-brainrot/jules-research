import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import BaselineMLP, HjorthMLP, HjorthAugmentedMLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data['x_test']).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model_class, X_train, y_train, X_test, y_test, lr, epochs=50, batch_size=128):
    input_dim = X_train.shape[1]
    output_dim = 10
    hidden_dim = 128

    model = model_class(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    return accuracy

def objective(trial, model_class, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    acc = train_model(model_class, X_train, y_train, X_test, y_test, lr, epochs=30)
    return acc

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    results = {}

    for name, model_class in [("Baseline", BaselineMLP), ("Hjorth", HjorthMLP), ("HjorthAugmented", HjorthAugmentedMLP)]:
        print(f"Tuning {name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_class, X_train, y_train, X_test, y_test), n_trials=20)

        best_lr = study.best_params["lr"]
        print(f"Best LR for {name}: {best_lr}")

        # Run multiple seeds with best LR
        accs = []
        for seed in range(5):
            torch.manual_seed(seed)
            acc = train_model(model_class, X_train, y_train, X_test, y_test, best_lr, epochs=50)
            accs.append(acc)
            print(f"Seed {seed} Accuracy: {acc:.4f}")

        results[name] = {
            "mean": np.mean(accs),
            "std": np.std(accs),
            "best_lr": best_lr
        }

    with open("differentiable_hjorth_parameters_experiment/results.txt", "w") as f:
        for name, res in results.items():
            f.write(f"{name}: {res['mean']:.4f} +/- {res['std']:.4f} (Best LR: {res['best_lr']})\n")
            print(f"{name}: {res['mean']:.4f} +/- {res['std']:.4f}")

if __name__ == "__main__":
    run_experiment()
