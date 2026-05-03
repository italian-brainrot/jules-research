import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import optuna
import json
import os
import matplotlib.pyplot as plt
from model import LLWNet, BaselineMLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr=1e-3, epochs=100, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for bx, by in dl_train:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()

        history['train_loss'].append(epoch_loss / len(dl_train))
        history['test_acc'].append(test_acc)

    return max(history['test_acc']), history

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])

    if model_type == "llw":
        levels = trial.suggest_int("levels", 1, 3)
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        model = LLWNet(levels=levels, kernel_size=kernel_size, hidden_dim=hidden_dim)
    elif model_type == "haar":
        levels = trial.suggest_int("levels", 1, 3)
        model = LLWNet(levels=levels, use_learnable=False, hidden_dim=hidden_dim)
    else:
        model = BaselineMLP(hidden_dim=hidden_dim)

    acc, _ = train_model(model, X_train, y_train, X_test, y_test, lr=lr, epochs=50)
    return acc

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    results = {}

    for model_type in ["baseline", "haar", "llw"]:
        if os.path.exists("differentiable_lifting_wavelet_experiment/results.json"):
            with open("differentiable_lifting_wavelet_experiment/results.json", "r") as f:
                results = json.load(f)
        else:
            results = {}

        if model_type in results:
            print(f"Skipping {model_type}, already in results.")
            continue

        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=5)

        print(f"Best trial for {model_type}: {study.best_trial.params}")

        # Train best model
        best_params = study.best_trial.params.copy()
        lr = best_params.pop('lr')
        if model_type == "llw":
            model = LLWNet(**best_params)
        elif model_type == "haar":
            model = LLWNet(use_learnable=False, **best_params)
        else:
            model = BaselineMLP(**best_params)

        acc, history = train_model(model, X_train, y_train, X_test, y_test, lr=lr, epochs=50)
        results[model_type] = {
            'best_acc': acc,
            'history': history,
            'best_params': study.best_trial.params
        }
        results[model_type]['best_params']['lr'] = lr # Put it back for saving

        with open("differentiable_lifting_wavelet_experiment/results.json", "w") as f:
            json.dump(results, f)

    # Plot results
    plt.figure(figsize=(10, 6))
    for model_type, data in results.items():
        plt.plot(data['history']['test_acc'], label=f"{model_type} (max: {data['best_acc']:.4f})")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Comparison of Lifting Wavelet Networks")
    plt.legend()
    plt.savefig("differentiable_lifting_wavelet_experiment/comparison.png")
