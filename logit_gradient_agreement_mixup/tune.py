import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from logit_gradient_agreement_mixup.model import MLP
from logit_gradient_agreement_mixup.data import get_data, get_loaders
from logit_gradient_agreement_mixup.train_utils import mixup_data, lgam_mixup_data, mixup_criterion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, loader, optimizer, criterion, mode, alpha=1.0, gamma=1.0):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if mode == 'Mixup':
            mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha)
            output = model(mixed_x)
            loss = mixup_criterion(criterion, output, y_a, y_b, lam)
        elif mode == 'LGAM':
            mixed_x, y_a, y_b, lams = lgam_mixup_data(model, x, y, alpha, gamma)
            output = model(mixed_x)
            loss = mixup_criterion(criterion, output, y_a, y_b, lams)
        else: # Baseline
            output = model(x)
            loss = criterion(output, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def objective(trial, mode, data):
    (X_train, y_train), (X_val, y_val), _ = data
    train_loader, val_loader, _ = get_loaders(X_train, y_train, X_val, y_val, X_val, y_val, batch_size=128)

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    alpha = 0
    gamma = 0
    if mode in ['Mixup', 'LGAM']:
        alpha = trial.suggest_float('alpha', 0.1, 2.0)
        if mode == 'LGAM':
            gamma = trial.suggest_float('gamma', 0.1, 5.0)

    model = MLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    epochs = 15
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, mode, alpha, gamma)

    val_acc = evaluate(model, val_loader)
    return val_acc

def tune_all():
    data = get_data()
    modes = ['Baseline', 'Mixup', 'LGAM']
    best_params = {}

    for mode in modes:
        print(f"Tuning {mode}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode, data), n_trials=20)
        best_params[mode] = study.best_params
        print(f"Best params for {mode}: {study.best_params}")

    return best_params

if __name__ == "__main__":
    tune_all()
