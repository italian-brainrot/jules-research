import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from householder_product_networks.models import get_model, count_parameters
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_model(model, dl_train, dl_test, lr, epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in dl_test:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc

    return best_acc

def objective(trial, model_name):
    X_train, y_train, X_test, y_test = get_data()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    input_size = 40
    hidden_size = 64
    output_size = 10

    if model_name == "householder":
        model = get_model(model_name, input_size, hidden_size, output_size, num_layers=2, num_reflectors=32)
    else:
        model = get_model(model_name, input_size, hidden_size, output_size, num_layers=2)

    return train_model(model, dl_train, dl_test, lr, epochs=20)

if __name__ == "__main__":
    for model_name in ["baseline", "householder"]:
        print(f"Tuning {model_name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_name), n_trials=10)

        print(f"Best {model_name} trial:")
        print(f"  Value: {study.best_trial.value}")
        print(f"  Params: {study.best_trial.params}")

        with open(f"householder_product_networks/best_params_{model_name}.txt", "w") as f:
            f.write(str(study.best_trial.params))
