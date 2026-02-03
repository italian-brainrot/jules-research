import torch
from torch import nn
import optuna
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from sgo_lib import MLP, get_sgo_penalty
import numpy as np
import os
import matplotlib.pyplot as plt

def get_data():
    args = get_dataset_args()
    args.num_samples = 4000
    data = make_dataset(args)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_model(mode, lr, lambda_reg, epochs=50, return_history=False):
    X_train, y_train, X_test, y_test = get_data()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_test_acc = 0.0
    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, y in dl_train:
            optimizer.zero_grad()

            logits = model(x)
            loss_ce = nn.functional.cross_entropy(logits, y)

            loss = loss_ce
            if mode == 'SGO':
                params = dict(model.named_parameters())
                penalty = get_sgo_penalty(params, model, x, y, class_aware=False)
                loss = loss + lambda_reg * penalty
            elif mode == 'CSGO':
                params = dict(model.named_parameters())
                penalty = get_sgo_penalty(params, model, x, y, class_aware=True)
                loss = loss + lambda_reg * penalty

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_preds = test_logits.argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()
            if test_acc > best_test_acc:
                best_test_acc = test_acc

        history['train_loss'].append(epoch_loss / len(dl_train))
        history['test_acc'].append(test_acc)

    if return_history:
        return best_test_acc, history
    return best_test_acc

def objective(trial, mode):
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    lambda_reg = 0.0
    if mode != 'Baseline':
        lambda_reg = trial.suggest_float('lambda_reg', 1e-3, 1.0, log=True)

    return train_model(mode, lr, lambda_reg, epochs=30)

if __name__ == "__main__":
    results = {}
    best_histories = {}

    modes = ['Baseline', 'SGO', 'CSGO']

    for mode in modes:
        print(f"--- Tuning {mode} ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, mode), n_trials=10)

        best_lr = study.best_params['lr']
        best_lambda = study.best_params.get('lambda_reg', 0.0)

        print(f"Best {mode} params: {study.best_params}")

        # Final run with more epochs
        print(f"Final run for {mode}...")
        best_acc, history = train_model(mode, best_lr, best_lambda, epochs=60, return_history=True)

        results[mode] = {
            'best_acc': best_acc,
            'params': study.best_params
        }
        best_histories[mode] = history

    # Save results
    with open('results.txt', 'w') as f:
        for mode, res in results.items():
            f.write(f"Mode: {mode}\n")
            f.write(f"  Best Acc: {res['best_acc']:.4f}\n")
            f.write(f"  Final Train Loss: {best_histories[mode]['train_loss'][-1]:.4f}\n")
            f.write(f"  Final Test Acc: {best_histories[mode]['test_acc'][-1]:.4f}\n")
            f.write(f"  Params: {res['params']}\n")
            f.write("-" * 20 + "\n")

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for mode in modes:
        plt.plot(best_histories[mode]['train_loss'], label=mode)
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for mode in modes:
        plt.plot(best_histories[mode]['test_acc'], label=mode)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('comparison.png')
    print("Done! Results saved to results.txt and comparison.png")
