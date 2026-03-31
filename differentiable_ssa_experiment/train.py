import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from ssa import SSANet
import matplotlib.pyplot as plt

def train_model(model, dl_train, dl_test, lr, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return 100 * correct / total

class BaselineMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def objective(trial):
    # Dataset
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)

    X_train = torch.tensor(data['x'])
    y_train = torch.tensor(data['y'])
    X_test = torch.tensor(data['x_test'])
    y_test = torch.tensor(data['y_test'])

    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=64, shuffle=False)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_size = 128

    model_type = trial.suggest_categorical("model_type", ["baseline", "ssa"])

    if model_type == "baseline":
        model = BaselineMLP(40, hidden_size, 10)
    else:
        window_size = trial.suggest_int("window_size", 5, 20)
        model = SSANet(40, window_size, hidden_size, 10)

    acc = train_model(model, dl_train, dl_test, lr)
    return acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best trials:")
    print(study.best_params)

    # Run comparison with best params over multiple seeds
    # ... (to be implemented in next step)
