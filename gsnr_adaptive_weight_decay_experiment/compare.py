import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mnist1d
import optuna
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from optimizer import GAWD
import os
import random

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data():
    args = mnist1d.get_dataset_args()
    data = mnist1d.get_dataset(args, path='gsnr_adaptive_weight_decay_experiment/mnist1d_data.pkl', download=True)
    x_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    x_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    # Split train into train and val
    n_train = int(0.8 * len(x_train))
    x_val = x_train[n_train:]
    y_val = y_train[n_train:]
    x_train = x_train[:n_train]
    y_train = y_train[:n_train]

    return x_train, y_train, x_val, y_val, x_test, y_test

class MLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    return total_loss / len(loader), correct / loader.data_length()

def objective(trial, optimizer_name, data):
    set_seed(42)
    x_train, y_train, x_val, y_val = data
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)

    model = MLP().to(device)
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)

    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        gamma = trial.suggest_float('gamma', 0.1, 10.0, log=True)
        optimizer = GAWD(model.parameters(), lr=lr, weight_decay=wd, gamma=gamma)

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(20):
        train_epoch(model, train_loader, optimizer, criterion)
        _, val_acc = evaluate(model, val_loader, criterion)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def run_experiment():
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)
    test_loader = TensorDataLoader((x_test, y_test), batch_size=128, shuffle=False)

    optimizers = ['AdamW', 'GAWD']
    best_params = {}

    data_for_obj = (x_train, y_train, x_val, y_val)
    for opt_name in optimizers:
        print(f"Optimizing {opt_name}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, opt_name, data_for_obj), n_trials=30)
        best_params[opt_name] = study.best_params
        print(f"Best params for {opt_name}: {study.best_params}")

    # Final training
    results = {}
    for opt_name in optimizers:
        print(f"Final training for {opt_name}...")
        set_seed(42)
        model = MLP().to(device)
        params = best_params[opt_name]
        if opt_name == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), **params)
        else:
            optimizer = GAWD(model.parameters(), **params)

        criterion = nn.CrossEntropyLoss()

        train_losses = []
        val_accs = []

        for epoch in range(50):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            _, val_acc = evaluate(model, val_loader, criterion)
            train_losses.append(train_loss)
            val_accs.append(val_acc)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

        test_loss, test_acc = evaluate(model, test_loader, criterion)
        results[opt_name] = {
            'train_losses': train_losses,
            'val_accs': val_accs,
            'test_acc': test_acc
        }
        print(f"{opt_name} Test Acc: {test_acc:.4f}")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for opt_name in optimizers:
        plt.plot(results[opt_name]['train_losses'], label=opt_name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for opt_name in optimizers:
        plt.plot(results[opt_name]['val_accs'], label=opt_name)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('gsnr_adaptive_weight_decay_experiment/results.png')

    # Save text results
    with open('gsnr_adaptive_weight_decay_experiment/results.txt', 'w') as f:
        for opt_name in optimizers:
            f.write(f"{opt_name} Best Params: {best_params[opt_name]}\n")
            f.write(f"{opt_name} Test Acc: {results[opt_name]['test_acc']:.4f}\n")

if __name__ == '__main__':
    run_experiment()
