import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import numpy as np
import os
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from differentiable_cepstrum_experiment.model import BaselineNet, CepstrumNet, CepstrumAugmentedNet

def get_data():
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    return X_train, y_train, X_test, y_test

def train_model(model, train_loader, val_loader, lr, epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x, y = batch
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                # Handle both tuple and list from TensorDataLoader
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                else:
                    x, y = batch
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        if total == 0:
            print(f"Warning: total is 0. len(val_loader)={len(val_loader)}")
            val_acc = 0
        else:
            val_acc = correct / total
        history['val_acc'].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc, history

def objective(trial, model_class, X_train, y_train, X_val, y_val):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    hidden_dim = 128

    input_dim = X_train.shape[1]
    output_dim = 10

    model = model_class(input_dim, hidden_dim, output_dim)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    val_loader = TensorDataLoader((X_val, y_val), batch_size=64, shuffle=False)

    acc, _ = train_model(model, train_loader, val_loader, lr, epochs=10)
    return acc

def main():
    torch.set_num_threads(1)
    X_train, y_train, X_test, y_test = get_data()

    # Use part of train as validation for Optuna
    # Ensure val_idx is NOT empty
    num_train_samples = X_train.shape[0]
    indices = torch.randperm(num_train_samples)
    train_split = int(0.8 * num_train_samples)
    train_idx = indices[:train_split]
    val_idx = indices[train_split:]
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # Check for empty data
    print(f"X_tr shape: {X_tr.shape}, X_val shape: {X_val.shape}")

    results = {}
    models = {
        'Baseline': BaselineNet,
        'Cepstrum': CepstrumNet,
        'CepstrumAugmented': CepstrumAugmentedNet
    }

    for name, model_class in models.items():
        print(f"Optimizing {name}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, model_class, X_tr, y_tr, X_val, y_val), n_trials=5)

        best_params = study.best_params
        if not best_params:
            print(f"Study for {name} returned no best params. Using default LR 0.001")
            best_lr = 0.001
        else:
            best_lr = best_params['lr']
            print(f"Best LR for {name}: {best_lr}")

        # Final training with best LR
        final_model = model_class(X_train.shape[1], 128, 10)
        train_loader = TensorDataLoader((X_train, y_train), batch_size=100, shuffle=True)
        test_loader = TensorDataLoader((X_test, y_test), batch_size=100, shuffle=False)

        acc, history = train_model(final_model, train_loader, test_loader, best_lr, epochs=30)
        results[name] = {
            'best_acc': acc,
            'history': history,
            'best_lr': best_lr
        }
        print(f"Final Test Accuracy for {name}: {acc:.4f}")

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data['history']['train_loss'], label=name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data['history']['val_acc'], label=name)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('differentiable_cepstrum_experiment/comparison.png')

    # Save textual summary
    with open('differentiable_cepstrum_experiment/results.txt', 'w') as f:
        for name, data in results.items():
            f.write(f"Model: {name}\n")
            f.write(f"Best LR: {data['best_lr']:.6f}\n")
            f.write(f"Best Test Accuracy: {data['best_acc']:.4f}\n")
            f.write("-" * 20 + "\n")

if __name__ == "__main__":
    main()
