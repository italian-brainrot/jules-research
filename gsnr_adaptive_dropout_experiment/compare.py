import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import optuna
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import GADMLP, BaselineMLP
import os

# Set device
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

def get_gsnr(optimizer, params):
    gsnrs = []
    for p in params:
        state = optimizer.state[p]
        if 'exp_avg' in state and 'exp_avg_sq' in state:
            m = state['exp_avg']
            v = state['exp_avg_sq']
            t = state['step']
            if isinstance(t, torch.Tensor):
                t = t.item()

            beta1, beta2 = optimizer.param_groups[0]['betas']

            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            gsnr = (m_hat.pow(2) / (v_hat + 1e-8))
            gsnrs.append(gsnr.mean().item())
    if not gsnrs:
        return 1.0 # Default to 1 (no dropout) if no state yet
    return np.mean(gsnrs)

def train_epoch(model, loader, optimizer, criterion, is_gad=False, gad_params=None):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if is_gad:
            # Update dropout rates based on current GSNR
            p_base, gamma = gad_params
            with torch.no_grad():
                gsnr1 = get_gsnr(optimizer, [model.fc1.weight, model.fc1.bias])
                gsnr2 = get_gsnr(optimizer, [model.fc2.weight, model.fc2.bias])

                # p_eff = p_base * (1 - GSNR)^gamma
                # clamp GSNR to [0, 1] just in case
                gsnr1 = max(0, min(1, gsnr1))
                gsnr2 = max(0, min(1, gsnr2))

                p1 = p_base * ((1 - gsnr1) ** gamma)
                p2 = p_base * ((1 - gsnr2) ** gamma)
                model.set_dropout_rates(p1, p2)

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

def objective(trial, mode, data):
    set_seed(42)
    x_train, y_train, x_val, y_val = data
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    if mode == 'Baseline':
        p = trial.suggest_float('p', 0.0, 0.7)
        model = BaselineMLP(p=p).to(device)
        gad_params = None
    else:
        p_base = trial.suggest_float('p_base', 0.0, 0.9)
        gamma = trial.suggest_float('gamma', 0.1, 5.0, log=True)
        model = GADMLP().to(device)
        gad_params = (p_base, gamma)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(30):
        train_epoch(model, train_loader, optimizer, criterion, is_gad=(mode=='GAD'), gad_params=gad_params)
        _, val_acc = evaluate(model, val_loader, criterion)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def run_experiment():
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    data_for_obj = (x_train, y_train, x_val, y_val)

    modes = ['Baseline', 'GAD']
    best_params = {}

    for mode in modes:
        print(f"Optimizing {mode}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode, data_for_obj), n_trials=30)
        best_params[mode] = study.best_params
        print(f"Best params for {mode}: {study.best_params}")

    # Final training
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)
    test_loader = TensorDataLoader((x_test, y_test), batch_size=128, shuffle=False)

    results = {}
    for mode in modes:
        print(f"Final training for {mode}...")
        set_seed(42)
        params = best_params[mode]

        if mode == 'Baseline':
            model = BaselineMLP(p=params['p']).to(device)
            gad_params = None
        else:
            model = GADMLP().to(device)
            gad_params = (params['p_base'], params['gamma'])

        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        criterion = nn.CrossEntropyLoss()

        history = {'train_loss': [], 'val_acc': []}
        best_val_acc = 0
        test_acc_at_best_val = 0

        for epoch in range(100):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, is_gad=(mode=='GAD'), gad_params=gad_params)
            _, val_acc = evaluate(model, val_loader, criterion)
            _, test_acc = evaluate(model, test_loader, criterion)

            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc_at_best_val = test_acc

            if (epoch+1) % 20 == 0:
                print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

        results[mode] = {
            'history': history,
            'test_acc': test_acc_at_best_val,
            'best_params': params
        }

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for mode in modes:
        plt.plot(results[mode]['history']['train_loss'], label=mode)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for mode in modes:
        plt.plot(results[mode]['history']['val_acc'], label=mode)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('gsnr_adaptive_dropout_experiment/results.png')

    # Save results to file
    with open('gsnr_adaptive_dropout_experiment/results.txt', 'w') as f:
        for mode in modes:
            f.write(f"Mode: {mode}\n")
            f.write(f"Best Params: {results[mode]['best_params']}\n")
            f.write(f"Test Acc (at best val): {results[mode]['test_acc']:.4f}\n")
            f.write("-" * 20 + "\n")

if __name__ == '__main__':
    run_experiment()
