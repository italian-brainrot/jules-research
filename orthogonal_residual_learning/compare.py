import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import os
import sys

# Add current directory to path to import models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import ResMLP

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_and_eval(model, train_loader, test_loader, optimizer, epochs, lambda_orth=0.0):
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        total_orth_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            ce_loss = criterion(outputs, targets)
            orth_loss = model.get_orth_loss()
            loss = ce_loss + lambda_orth * orth_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total_orth_loss += orth_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100. * correct / total
        avg_cos_sim = model.get_avg_cos_sim()
        avg_cos_sim_sq = model.get_avg_cos_sim_sq()

        history.append({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'test_loss': test_loss / len(test_loader),
            'test_acc': test_acc,
            'orth_loss': total_orth_loss / len(train_loader),
            'avg_cos_sim': avg_cos_sim,
            'avg_cos_sim_sq': avg_cos_sim_sq
        })

    return history

def objective(trial, variant, X_train, y_train, X_test, y_test):
    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    lambda_orth = 0.0
    if variant == 'penalty':
        lambda_orth = trial.suggest_float('lambda_orth', 1e-4, 10.0, log=True)

    model = ResMLP(input_dim=40, hidden_dim=128, output_dim=10, num_blocks=4, variant=variant).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Fast evaluation: 15 epochs
    history = train_and_eval(model, train_loader, test_loader, optimizer, epochs=15, lambda_orth=lambda_orth)
    return max(h['test_acc'] for h in history)

def run_experiment():
    variants = ['baseline', 'penalty', 'forced']
    best_params = {}

    X_train, y_train, X_test, y_test = get_data()

    for variant in variants:
        print(f"Tuning {variant}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, variant, X_train, y_train, X_test, y_test), n_trials=15)
        best_params[variant] = study.best_params
        print(f"Best params for {variant}: {study.best_params}")

    # Final training
    results = {}
    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    for variant in variants:
        print(f"Final training for {variant}...")
        params = best_params[variant]
        model = ResMLP(input_dim=40, hidden_dim=128, output_dim=10, num_blocks=4, variant=variant).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        lambda_orth = params.get('lambda_orth', 0.0)

        history = train_and_eval(model, train_loader, test_loader, optimizer, epochs=50, lambda_orth=lambda_orth)
        results[variant] = history

    # Plotting
    plt.figure(figsize=(15, 10))

    # Plot Test Accuracy
    plt.subplot(2, 2, 1)
    for variant in variants:
        acc = [h['test_acc'] for h in results[variant]]
        plt.plot(acc, label=variant)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Plot Average Cosine Similarity
    plt.subplot(2, 2, 2)
    for variant in variants:
        sim = [h['avg_cos_sim'] for h in results[variant]]
        plt.plot(sim, label=variant)
    plt.title('Average Cosine Similarity (x, residual)')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()

    # Plot Average Cosine Similarity Squared
    plt.subplot(2, 2, 3)
    for variant in variants:
        sim_sq = [h['avg_cos_sim_sq'] for h in results[variant]]
        plt.plot(sim_sq, label=variant)
    plt.title('Average Cosine Similarity Squared')
    plt.xlabel('Epoch')
    plt.ylabel('Cos Sim Squared')
    plt.legend()

    # Plot Test Loss
    plt.subplot(2, 2, 4)
    for variant in variants:
        loss = [h['test_loss'] for h in results[variant]]
        plt.plot(loss, label=variant)
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('orthogonal_residual_learning/results.png')

    # Save results to CSV
    for variant in variants:
        pd.DataFrame(results[variant]).to_csv(f'orthogonal_residual_learning/{variant}_history.csv', index=False)

    # Write a summary
    with open('orthogonal_residual_learning/summary.txt', 'w') as f:
        for variant in variants:
            best_acc = max(h['test_acc'] for h in results[variant])
            f.write(f"{variant}: Best Test Acc = {best_acc:.2f}%, Params = {best_params[variant]}\n")

if __name__ == "__main__":
    run_experiment()
