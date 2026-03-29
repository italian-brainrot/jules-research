import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import VolterraMLP, BaselineMLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test'])
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, lr, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_test).float().mean().item()
    return accuracy

def objective(trial, model_type):
    X_train, y_train, X_test, y_test = get_data()
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_type == "volterra":
        model = VolterraMLP(hidden_dim=40)
    else:
        model = BaselineMLP(hidden_dim=256)

    train_model(model, X_train, y_train, lr, epochs=15)
    acc = evaluate_model(model, X_test, y_test)
    return acc

if __name__ == "__main__":
    print("Tuning VolterraMLP...")
    study_volterra = optuna.create_study(direction="maximize")
    study_volterra.optimize(lambda t: objective(t, "volterra"), n_trials=10)
    print(f"Best LR for VolterraMLP: {study_volterra.best_params['lr']}, Acc: {study_volterra.best_value}")

    print("\nTuning BaselineMLP...")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(lambda t: objective(t, "baseline"), n_trials=10)
    print(f"Best LR for BaselineMLP: {study_baseline.best_params['lr']}, Acc: {study_baseline.best_value}")

    with open("volterra_filter_network/best_params.txt", "w") as f:
        f.write(f"volterra_lr: {study_volterra.best_params['lr']}\n")
        f.write(f"baseline_lr: {study_baseline.best_params['lr']}\n")
