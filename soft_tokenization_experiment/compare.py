import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import os
import numpy as np
from light_dataloader import TensorDataLoader
from sklearn.datasets import fetch_20newsgroups
from soft_tokenization_experiment.model import TokenizationModel

# Set seed for reproducibility
torch.manual_seed(42)

def get_data(seq_len=128):
    categories = ['alt.atheism', 'sci.space', 'comp.graphics', 'talk.politics.mideast']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

    def preprocess(texts, targets):
        X = []
        y = []
        for text, target in zip(texts, targets):
            if len(text) < seq_len:
                continue
            # Character level encoding
            encoded = [ord(c) if ord(c) < 256 else 0 for c in text[:seq_len]]
            X.append(encoded)
            y.append(target)
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    X_train, y_train = preprocess(newsgroups_train.data, newsgroups_train.target)
    X_test, y_test = preprocess(newsgroups_test.data, newsgroups_test.target)

    return (X_train, y_train), (X_test, y_test)

def get_model(method):
    return TokenizationModel(
        vocab_size=256,
        input_len=128,
        output_len=32,
        dim=64,
        nhead=4,
        num_layers=2,
        method=method
    )

def train_model(model, train_loader, test_loader, lr, epochs=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        history['train_loss'].append(total_loss / len(train_loader))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        history['test_acc'].append(correct / total)

    return history

def objective(trial, method, train_data):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    model = get_model(method)

    X_train, y_train = train_data
    # Use a subset for faster tuning
    n_tune = min(1000, len(X_train))
    train_loader = TensorDataLoader((X_train[:int(n_tune*0.7)], y_train[:int(n_tune*0.7)]), batch_size=64, shuffle=True)
    val_loader = TensorDataLoader((X_train[int(n_tune*0.7):n_tune], y_train[int(n_tune*0.7):n_tune]), batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(15):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = get_data()
    print(f"Dataset sizes: Train {len(X_train)}, Test {len(X_test)}")
    train_loader = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=64)

    methods = ["Uniform", "Attention", "BGST"]
    results = {}
    best_lrs = {}

    for method in methods:
        print(f"Tuning {method}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, method, (X_train, y_train)), n_trials=8)
        best_lr = study.best_params['lr']
        best_lrs[method] = best_lr
        print(f"Best LR for {method}: {best_lr}")

        print(f"Training {method} with best LR...")
        model = get_model(method)
        history = train_model(model, train_loader, test_loader, best_lr, epochs=30)
        results[method] = history
        print(f"{method} Final Test Acc: {history['test_acc'][-1]:.4f}")

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for method, history in results.items():
        plt.plot(history['train_loss'], label=method)
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for method, history in results.items():
        plt.plot(history['test_acc'], label=method)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('soft_tokenization_experiment/comparison.png')

    # Save summary
    with open('soft_tokenization_experiment/results.txt', 'w') as f:
        for method in methods:
            f.write(f"{method}: Best LR={best_lrs[method]:.6f}, Final Acc={results[method]['test_acc'][-1]:.4f}\n")

    print("Experiment complete. Results saved.")
