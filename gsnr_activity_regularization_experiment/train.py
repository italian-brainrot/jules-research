import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import GWARMLP, BaselineMLP
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data():
    args = get_dataset_args()
    args.num_samples = 8000
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

def train_epoch(model, loader, optimizer, criterion, mode, lambda_gwar=0.0):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        ce_loss = criterion(output, y)

        if mode == 'GWAR':
            gwar_loss = model.compute_gwar_loss(y, lambda_gwar)
            loss = ce_loss + gwar_loss
        else:
            loss = ce_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    return correct / loader.data_length()

def objective(trial, mode, data):
    set_seed(42)
    x_train, y_train, x_val, y_val = data
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)

    if mode == 'GWAR':
        lambda_gwar = trial.suggest_float('lambda_gwar', 1e-5, 1.0, log=True)
        model = GWARMLP().to(device)
    else:
        lambda_gwar = 0.0
        model = BaselineMLP().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(20): # Short optimization
        train_epoch(model, train_loader, optimizer, criterion, mode, lambda_gwar)
        val_acc = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def run_experiment():
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    data_for_obj = (x_train, y_train, x_val, y_val)

    modes = ['Baseline', 'GWAR']
    best_params = {}

    for mode in modes:
        print(f"Optimizing {mode}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode, data_for_obj), n_trials=30)
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

        for seed in seeds:
            set_seed(seed)
            if mode == 'GWAR':
                model = GWARMLP().to(device)
                lambda_gwar = params['lambda_gwar']
            else:
                model = BaselineMLP().to(device)
                lambda_gwar = 0.0

            optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0
            test_acc_at_best_val = 0

            for epoch in range(50):
                train_epoch(model, train_loader, optimizer, criterion, mode, lambda_gwar)
                val_acc = evaluate(model, val_loader)
                test_acc = evaluate(model, test_loader)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc_at_best_val = test_acc

            mode_results.append(test_acc_at_best_val)
            print(f"Seed {seed}, Test Acc: {test_acc_at_best_val:.4f}")

        results[mode] = {
            'test_accs': mode_results,
            'mean': np.mean(mode_results),
            'std': np.std(mode_results),
            'best_params': params
        }

    # Save results to README.md
    with open('gsnr_activity_regularization_experiment/README.md', 'w') as f:
        f.write("# GSNR-based Activity Regularization (GWAR) Experiment\n\n")
        f.write("## Hypothesis\n")
        f.write("Penalizing neurons that receive inconsistent gradient signals across a batch encourages the network to rely on more robust features. ")
        f.write("The GSNR of the activation gradients is used to weight the activity regularization penalty: ")
        f.write("$L_{GWAR} = \\lambda \\sum_{layer} \\sum_{i} (1 - GSNR_i) \\cdot \\text{mean}_b(a_{b,i}^2)$\n\n")

        f.write("## Results\n")
        f.write("| Mode | Test Accuracy | Best Hyperparameters |\n")
        f.write("| --- | --- | --- |\n")
        for mode in modes:
            res = results[mode]
            f.write(f"| {mode} | {res['mean']:.4f} ± {res['std']:.4f} | {res['best_params']} |\n")

        f.write("\n## Discussion\n")
        if results['GWAR']['mean'] > results['Baseline']['mean']:
            f.write("GWAR outperformed the baseline, suggesting that gradient-consistency-based activity regularization is beneficial.\n")
        else:
            f.write("GWAR did not outperform the baseline in this setup.\n")

    print("Experiment completed. Results saved to gsnr_activity_regularization_experiment/README.md")

if __name__ == '__main__':
    run_experiment()
