import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import os
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from vmd import DVMD
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class BaselineMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x.float())

class DVMDNet(nn.Module):
    def __init__(self, n_modes, hidden_size, num_classes, alpha=2000, n_iter=20):
        super().__init__()
        self.vmd = DVMD(n_modes=n_modes, alpha=alpha, n_iter=n_iter)
        # Features are energies (K) and frequencies (K)
        self.fc = nn.Sequential(
            nn.Linear(2 * n_modes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        energies, omegas = self.vmd(x.float())
        features = torch.cat([energies, omegas], dim=1)
        # Normalize features roughly
        features = torch.log1p(features)
        return self.fc(features)

class DVMDAugmentedMLP(nn.Module):
    def __init__(self, input_size, n_modes, hidden_size, num_classes, alpha=2000, n_iter=20):
        super().__init__()
        self.vmd = DVMD(n_modes=n_modes, alpha=alpha, n_iter=n_iter)
        self.fc = nn.Sequential(
            nn.Linear(input_size + 2 * n_modes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = x.float()
        energies, omegas = self.vmd(x)
        vmd_features = torch.cat([energies, omegas], dim=1)
        vmd_features = torch.log1p(vmd_features)
        combined = torch.cat([x, vmd_features], dim=1)
        return self.fc(combined)

def train_model(model, dl_train, dl_test, lr, epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
                _, predicted = torch.max(out.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc

    return best_acc

if __name__ == "__main__":
    args = get_dataset_args()
    args.num_samples = 5000
    data = make_dataset(args)

    X_train = torch.tensor(data['x'])
    y_train = torch.tensor(data['y'])
    X_test = torch.tensor(data['x_test'])
    y_test = torch.tensor(data['y_test'])

    dl_train = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=256, shuffle=False)

    results = {}

    for m_type in ["baseline", "vmd_net", "vmd_augmented"]:
        print(f"Fine-tuning {m_type}...")
        m_study = optuna.create_study(direction="maximize")
        def m_obj(trial):
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            if m_type == "baseline":
                model = BaselineMLP(40, 128, 10)
            elif m_type == "vmd_net":
                n_modes = trial.suggest_int("n_modes", 2, 5)
                model = DVMDNet(n_modes, 128, 10, n_iter=10)
            else:
                n_modes = trial.suggest_int("n_modes", 2, 5)
                model = DVMDAugmentedMLP(40, n_modes, 128, 10, n_iter=10)
            return train_model(model, dl_train, dl_test, lr, epochs=20)

        m_study.optimize(m_obj, n_trials=5)

        print(f"Evaluating {m_type} with best params: {m_study.best_params}")
        accs = []
        for seed in range(3):
            torch.manual_seed(seed)
            if m_type == "baseline":
                model = BaselineMLP(40, 128, 10)
            elif m_type == "vmd_net":
                model = DVMDNet(m_study.best_params["n_modes"], 128, 10, n_iter=10)
            else:
                model = DVMDAugmentedMLP(40, m_study.best_params["n_modes"], 128, 10, n_iter=10)
            acc = train_model(model, dl_train, dl_test, m_study.best_params["lr"], epochs=30)
            accs.append(acc)

        results[m_type] = (np.mean(accs), np.std(accs))
        print(f"{m_type}: {results[m_type][0]:.2f}% +- {results[m_type][1]:.2f}%")

    with open("differentiable_vmd_experiment/results.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v[0]:.2f}% +- {v[1]:.2f}%\n")
