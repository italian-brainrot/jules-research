import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import os
import numpy as np
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset, get_dataset_args
from layers import MarkovStationaryPooling, AttentionPooling

# Set seed for reproducibility
torch.manual_seed(42)

def get_data():
    args = get_dataset_args()
    args.num_samples = 4000
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist1d_data.pkl')
    # Using download=True will create the file if it doesn't exist
    # If there's an issue with loading, we delete and retry
    try:
        data = get_dataset(args, path=data_path, download=True)
    except Exception:
        if os.path.exists(data_path):
            os.remove(data_path)
        data = get_dataset(args, path=data_path, download=True)

    X_train = torch.tensor(data['x'], dtype=torch.float32).unsqueeze(1) # [B, 1, 40]
    y_train = torch.tensor(data['y'], dtype=torch.int64)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(data['y_test'], dtype=torch.int64)

    return (X_train, y_train), (X_test, y_test)

class BaseConvModel(nn.Module):
    def __init__(self, pooling_layer, pooling_out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pooling = pooling_layer
        self.fc = nn.Linear(pooling_out_dim, 10)

    def forward(self, x):
        # x: [B, 1, 40]
        x = self.conv(x) # [B, 64, 40]
        x = x.transpose(1, 2) # [B, 40, 64] for pooling
        p = self.pooling(x)
        if isinstance(p, tuple):
            p = torch.cat(p, dim=-1)
        logits = self.fc(p)
        return logits

def get_model(name):
    if name == "GAP":
        pool = lambda x: torch.mean(x, dim=1)
        return BaseConvModel(pool, 64)
    elif name == "AttnP":
        pool = AttentionPooling(64)
        return BaseConvModel(pool, 64)
    elif name == "MSP":
        pool = MarkovStationaryPooling(64, num_iters=20, include_entropy=False)
        return BaseConvModel(pool, 64)
    elif name == "MSPE":
        pool = MarkovStationaryPooling(64, num_iters=20, include_entropy=True)
        return BaseConvModel(pool, 65) # 64 + 1
    else:
        raise ValueError(name)

def train_model(model, train_loader, test_loader, lr, epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
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
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        history['test_acc'].append(correct / total)

    return history

def objective(trial, name, train_data):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    model = get_model(name)

    X_train, y_train = train_data
    # Use a subset for faster tuning
    train_loader = TensorDataLoader((X_train[:1500], y_train[:1500]), batch_size=64, shuffle=True)
    val_loader = TensorDataLoader((X_train[1500:2000], y_train[1500:2000]), batch_size=64)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(15):
        model.train()
        for x, y in train_loader:
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
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = get_data()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=64)

    models_to_test = ["GAP", "AttnP", "MSP", "MSPE"]
    results = {}
    best_lrs = {}

    for name in models_to_test:
        print(f"Tuning {name}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, name, (X_train, y_train)), n_trials=10)
        best_lr = study.best_params['lr']
        best_lrs[name] = best_lr
        print(f"Best LR for {name}: {best_lr}")

        print(f"Training {name} with best LR...")
        model = get_model(name)
        history = train_model(model, train_loader, test_loader, best_lr, epochs=50)
        results[name] = history
        print(f"{name} Final Test Acc: {history['test_acc'][-1]:.4f}")

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, history in results.items():
        plt.plot(history['train_loss'], label=name)
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, history in results.items():
        plt.plot(history['test_acc'], label=name)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'results.png'))

    # Save summary
    summary_path = os.path.join(os.path.dirname(__file__), 'summary.txt')
    with open(summary_path, 'w') as f:
        for name in models_to_test:
            f.write(f"{name}: Best LR={best_lrs[name]:.6f}, Final Acc={results[name]['test_acc'][-1]:.4f}\n")

    print("Experiment complete. Results saved.")
