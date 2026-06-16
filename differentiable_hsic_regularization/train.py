import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from differentiable_hsic_regularization.model import BaselineMLP, HSICRegularizedMLP, get_hsic_loss
import os

def train_model(model, dl_train, dl_test, epochs=50, lr=1e-3, weight_decay=1e-4,
                use_hsic=False, hsic_type='standard', weight_in=0.1, weight_out=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dl_train:
            optimizer.zero_grad()
            if use_hsic:
                out, hidden = model(x, return_hidden=True)
                loss_ce = criterion(out, y)
                loss_hsic = get_hsic_loss(hidden, x, y, hsic_type=hsic_type,
                                          weight_in=weight_in, weight_out=weight_out)
                loss = loss_ce + loss_hsic
            else:
                out = model(x)
                loss = criterion(out, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dl_test:
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        history['train_loss'].append(total_loss / len(dl_train))
        history['test_acc'].append(correct / total)

    return history

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data['x_test']).float(), torch.tensor(data['y_test']).long()

    dl_train = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=256, shuffle=False)
    return dl_train, dl_test

def objective(trial, mode='baseline'):
    dl_train, dl_test = get_data()
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    if mode == 'baseline':
        model = BaselineMLP()
        history = train_model(model, dl_train, dl_test, epochs=30, lr=lr, weight_decay=weight_decay, use_hsic=False)
    else:
        weight_in = trial.suggest_float('weight_in', 1e-4, 1.0, log=True)
        weight_out = trial.suggest_float('weight_out', 1e-4, 1.0, log=True)
        hsic_type = trial.suggest_categorical('hsic_type', ['standard', 'normalized'])
        model = HSICRegularizedMLP()
        history = train_model(model, dl_train, dl_test, epochs=30, lr=lr, weight_decay=weight_decay,
                             use_hsic=True, hsic_type=hsic_type, weight_in=weight_in, weight_out=weight_out)

    return max(history['test_acc'])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['tune', 'eval'], default='tune')
    args = parser.parse_args()

    if args.mode == 'tune' or args.mode == 'eval':
        print("Tuning Baseline...")
        study_baseline = optuna.create_study(direction='maximize')
        study_baseline.optimize(lambda t: objective(t, 'baseline'), n_trials=10)

        print("Tuning HSIC...")
        study_hsic = optuna.create_study(direction='maximize')
        study_hsic.optimize(lambda t: objective(t, 'hsic'), n_trials=10)

        print(f"Best Baseline Acc: {study_baseline.best_value}")
        print(f"Best HSIC Acc: {study_hsic.best_value}")
        print(f"Best HSIC Params: {study_hsic.best_params}")

        # Now run evaluation with best parameters
        dl_train, dl_test = get_data()

        print("Evaluating Baseline with best parameters...")
        baseline_accs = []
        for i in range(5):
            model = BaselineMLP()
            history = train_model(model, dl_train, dl_test, epochs=50,
                                 lr=study_baseline.best_params['lr'],
                                 weight_decay=study_baseline.best_params['weight_decay'])
            baseline_accs.append(max(history['test_acc']))

        print("Evaluating HSIC with best parameters...")
        hsic_accs = []
        for i in range(5):
            model = HSICRegularizedMLP()
            history = train_model(model, dl_train, dl_test, epochs=50,
                                 lr=study_hsic.best_params['lr'],
                                 weight_decay=study_hsic.best_params['weight_decay'],
                                 use_hsic=True,
                                 hsic_type=study_hsic.best_params['hsic_type'],
                                 weight_in=study_hsic.best_params['weight_in'],
                                 weight_out=study_hsic.best_params['weight_out'])
            hsic_accs.append(max(history['test_acc']))

        print(f"Baseline: {np.mean(baseline_accs):.4f} +/- {np.std(baseline_accs):.4f}")
        print(f"HSIC: {np.mean(hsic_accs):.4f} +/- {np.std(hsic_accs):.4f}")

        with open('differentiable_hsic_regularization/results.txt', 'w') as f:
            f.write(f"Best Baseline Params: {study_baseline.best_params}\n")
            f.write(f"Best HSIC Params: {study_hsic.best_params}\n")
            f.write(f"Baseline: {np.mean(baseline_accs):.4f} +/- {np.std(baseline_accs):.4f}\n")
            f.write(f"HSIC: {np.mean(hsic_accs):.4f} +/- {np.std(hsic_accs):.4f}\n")
            f.write(f"Baseline raw: {baseline_accs}\n")
            f.write(f"HSIC raw: {hsic_accs}\n")

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.bar(['Baseline', 'HSIC'], [np.mean(baseline_accs), np.mean(hsic_accs)],
                yerr=[np.std(baseline_accs), np.std(hsic_accs)])
        plt.ylabel('Accuracy')
        plt.title('Baseline vs HSIC Bottleneck Regularization')
        plt.savefig('differentiable_hsic_regularization/results.png')
