import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from nca_loss import NCALoss

# Models
class BaselineMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class NCARegularizedMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embeddings = self.encoder(x)
        logits = self.classifier(embeddings)
        return logits, embeddings

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test'])
    return X_train, y_train, X_test, y_test

def train_model(model, dl_train, dl_test, epochs=40, lr=1e-3, nca_weight=0.0, nca_temp=1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    nca_criterion = NCALoss(temperature=nca_temp)

    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dl_train:
            optimizer.zero_grad()
            if isinstance(model, NCARegularizedMLP):
                logits, embeddings = model(x)
                ce_loss = criterion(logits, y)
                if nca_weight > 0:
                    reg_loss = nca_criterion(embeddings, y)
                    loss = ce_loss + nca_weight * reg_loss
                else:
                    loss = ce_loss
            else:
                logits = model(x)
                loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dl_test:
                if isinstance(model, NCARegularizedMLP):
                    logits, _ = model(x)
                else:
                    logits = model(x)
                _, predicted = torch.max(logits.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        acc = correct / total
        history['train_loss'].append(total_loss / len(dl_train))
        history['test_acc'].append(acc)

    return history

def objective_baseline(trial):
    X_train, y_train, X_test, y_test = get_data()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=64, shuffle=False)

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    model = BaselineMLP()
    history = train_model(model, dl_train, dl_test, epochs=15, lr=lr)
    return history['test_acc'][-1]

def objective_nca(trial):
    X_train, y_train, X_test, y_test = get_data()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=64, shuffle=False)

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    nca_weight = trial.suggest_float('nca_weight', 0.01, 10.0, log=True)
    nca_temp = trial.suggest_float('nca_temp', 0.1, 10.0, log=True)

    model = NCARegularizedMLP()
    history = train_model(model, dl_train, dl_test, epochs=15, lr=lr, nca_weight=nca_weight, nca_temp=nca_temp)
    return history['test_acc'][-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='eval', choices=['tune', 'eval'])
    args = parser.parse_args()

    if args.mode == 'tune':
        print("Tuning Baseline...")
        study_b = optuna.create_study(direction='maximize')
        study_b.optimize(objective_baseline, n_trials=10)

        print("Tuning NCA...")
        study_nca = optuna.create_study(direction='maximize')
        study_nca.optimize(objective_nca, n_trials=15)

        best_params = {
            'baseline': study_b.best_trial.params,
            'nca': study_nca.best_trial.params
        }
        print("Best params:", best_params)
        with open('best_params.json', 'w') as f:
            json.dump(best_params, f)

    else:
        X_train, y_train, X_test, y_test = get_data()
        dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
        dl_test = TensorDataLoader((X_test, y_test), batch_size=64, shuffle=False)

        if os.path.exists('best_params.json'):
             with open('best_params.json', 'r') as f:
                 best_params = json.load(f)
        else:
             best_params = {
                 'baseline': {'lr': 0.001},
                 'nca': {'lr': 0.001, 'nca_weight': 0.1, 'nca_temp': 1.0}
             }

        num_seeds = 3
        baseline_accs = []
        nca_accs = []

        plt.figure(figsize=(10, 5))

        for seed in range(num_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            print(f"Seed {seed}: Training Baseline...")
            model_b = BaselineMLP()
            history_b = train_model(model_b, dl_train, dl_test, epochs=40, lr=best_params['baseline']['lr'])
            baseline_accs.append(history_b['test_acc'][-1])

            print(f"Seed {seed}: Training NCA Regularized...")
            model_nca = NCARegularizedMLP()
            history_nca = train_model(model_nca, dl_train, dl_test, epochs=40,
                                     lr=best_params['nca']['lr'],
                                     nca_weight=best_params['nca']['nca_weight'],
                                     nca_temp=best_params['nca']['nca_temp'])
            nca_accs.append(history_nca['test_acc'][-1])

            if seed == 0:
                plt.plot(history_b['test_acc'], label='Baseline')
                plt.plot(history_nca['test_acc'], label='NCA Regularized')

        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.title('Baseline vs NCA Regularized MLP')
        plt.savefig('accuracy_comparison.png')

        results = f"Baseline Accuracy: {np.mean(baseline_accs):.4f} +/- {np.std(baseline_accs):.4f}\n"
        results += f"NCA Regularized Accuracy: {np.mean(nca_accs):.4f} +/- {np.std(nca_accs):.4f}\n"
        results += f"Best Baseline Params: {best_params['baseline']}\n"
        results += f"Best NCA Params: {best_params['nca']}\n"
        print(results)
        with open('results.txt', 'w') as f:
            f.write(results)
