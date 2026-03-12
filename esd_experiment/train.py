import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from esd_lib import DecorrManager
import os
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data():
    args = get_dataset_args()
    args.num_samples = 4000
    data = make_dataset(args)
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

def train_epoch(model, loader, optimizer, criterion, decorr_manager, lambda_decorr=0.0):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if decorr_manager:
            decorr_manager.clear()

        output = model(x)
        ce_loss = criterion(output, y)

        if decorr_manager and lambda_decorr > 0:
            d_loss = decorr_manager.compute_loss(ce_loss)
            loss = ce_loss + lambda_decorr * d_loss
        else:
            loss = ce_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_loss += criterion(output, y).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    return total_loss / len(loader), correct / loader.data_length()

def objective(trial, mode, data):
    set_seed(42)
    x_train, y_train, x_val, y_val = data
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)

    model = MLP().to(device)
    decorr_manager = None
    lambda_decorr = 0.0

    if mode != 'Baseline':
        lambda_decorr = trial.suggest_float('lambda_decorr', 1e-4, 10.0, log=True)
        decorr_manager = DecorrManager(model, mode=mode)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(20):
        train_epoch(model, train_loader, optimizer, criterion, decorr_manager, lambda_decorr)
        _, val_acc = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    if decorr_manager:
        decorr_manager.remove_hooks()

    return best_val_acc

def run_experiment():
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    data_for_obj = (x_train, y_train, x_val, y_val)

    modes = ['Baseline', 'Decorr', 'ESD']
    best_params = {}

    for mode in modes:
        print(f"Optimizing {mode}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode, data_for_obj), n_trials=20)
        best_params[mode] = study.best_params
        print(f"Best params for {mode}: {study.best_params}")

    # Final training and evaluation
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)
    test_loader = TensorDataLoader((x_test, y_test), batch_size=128, shuffle=False)

    results = {}
    seeds = [42, 43, 44]

    for mode in modes:
        mode_results = []
        print(f"Final training for {mode}...")
        params = best_params[mode]
        lambda_decorr = params.get('lambda_decorr', 0.0)

        for seed in seeds:
            set_seed(seed)
            model = MLP().to(device)
            decorr_manager = DecorrManager(model, mode=mode) if mode != 'Baseline' else None
            optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            criterion = nn.CrossEntropyLoss()

            history = {'train_loss': [], 'val_acc': [], 'test_acc': []}
            best_val_acc = 0
            test_acc_at_best_val = 0

            for epoch in range(50):
                train_loss = train_epoch(model, train_loader, optimizer, criterion, decorr_manager, lambda_decorr)
                val_loss, val_acc = evaluate(model, val_loader)
                test_loss, test_acc = evaluate(model, test_loader)

                history['train_loss'].append(train_loss)
                history['val_acc'].append(val_acc)
                history['test_acc'].append(test_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc_at_best_val = test_acc

            mode_results.append({
                'test_acc': test_acc_at_best_val,
                'history': history
            })
            print(f"Seed {seed}, Test Acc: {test_acc_at_best_val:.4f}")

            if decorr_manager:
                decorr_manager.remove_hooks()

        results[mode] = {
            'test_accs': [r['test_acc'] for r in mode_results],
            'mean': np.mean([r['test_acc'] for r in mode_results]),
            'std': np.std([r['test_acc'] for r in mode_results]),
            'best_params': params,
            'histories': [r['history'] for r in mode_results]
        }

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for mode in modes:
        avg_train_loss = np.mean([h['train_loss'] for h in results[mode]['histories']], axis=0)
        plt.plot(avg_train_loss, label=mode)
    plt.title('Average Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for mode in modes:
        avg_val_acc = np.mean([h['val_acc'] for h in results[mode]['histories']], axis=0)
        plt.plot(avg_val_acc, label=mode)
    plt.title('Average Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('esd_experiment/results.png')
    plt.close()

    # Save README
    save_readme(results, modes)

def save_readme(results, modes):
    with open('esd_experiment/README.md', 'w') as f:
        f.write("# Error Signal Decorrelation (ESD) Experiment\n\n")
        f.write("## Hypothesis\n")
        f.write("While standard activation decorrelation penalizes the correlation of neuron outputs, ")
        f.write("Error Signal Decorrelation (ESD) penalizes the correlation of the gradients of the loss with respect to those outputs (pre-activations). ")
        f.write("The hypothesis is that decorrelating error signals prevents neurons from being updated in redundant ways, ")
        f.write("potentially leading to more diverse and robust feature learning than standard decorrelation or a baseline MLP.\n\n")

        f.write("## Methodology\n")
        f.write("- **Dataset**: `mnist1d` (4,000 samples)\n")
        f.write("- **Model**: 3-layer MLP (40 -> 256 -> 256 -> 10)\n")
        f.write("- **Modes**:\n")
        f.write("  - **Baseline**: Standard AdamW optimizer.\n")
        f.write("  - **Decorr**: AdamW + Activation Decorrelation (squared correlation of pre-activations).\n")
        f.write("  - **ESD**: AdamW + Error Signal Decorrelation (squared correlation of gradients w.r.t. pre-activations).\n")
        f.write("- **Tuning**: Optuna was used for 20 trials per mode to find the best `lr`, `weight_decay`, and `lambda_decorr`.\n")
        f.write("- **Evaluation**: Best hyperparameters were used to train for 50 epochs over 3 random seeds.\n\n")

        f.write("## Results\n")
        f.write("| Mode | Test Accuracy | Best Hyperparameters |\n")
        f.write("| --- | --- | --- |\n")
        for mode in modes:
            res = results[mode]
            f.write(f"| {mode} | {res['mean']:.4f} ± {res['std']:.4f} | {res['best_params']} |\n")

        f.write("\n## Visualizations\n")
        f.write("### Training and Validation Curves\n")
        f.write("![Results](results.png)\n\n")

        f.write("## Discussion\n")
        best_mode = max(results, key=lambda k: results[k]['mean'])
        if best_mode == 'ESD':
            f.write("ESD outperformed both the baseline and standard decorrelation, supporting the hypothesis that decorrelating error signals is a more effective form of diversity-promoting regularization.\n")
        elif best_mode == 'Decorr':
            f.write("Standard activation decorrelation performed best, suggesting that decorrelating the features themselves is more important than decorrelating their update signals on this dataset.\n")
        else:
            f.write("The baseline AdamW performed best, indicating that the added decorrelation penalties (either activation or error signal) did not provide significant benefits for this specific model and dataset.\n")

if __name__ == '__main__':
    run_experiment()
