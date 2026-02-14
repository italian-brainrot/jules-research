import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import optuna
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import GRDMLP, NGRDMLP, BaselineMLP
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

def get_gsnr_per_param(optimizer, p):
    state = optimizer.state[p]
    if 'exp_avg' in state and 'exp_avg_sq' in state:
        m = state['exp_avg']
        v = state['exp_avg_sq']
        t = state.get('step', torch.tensor(0.0))
        if isinstance(t, torch.Tensor):
            t = t.item()

        if t == 0: return torch.ones_like(p)

        beta1, beta2 = optimizer.param_groups[0]['betas']

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        gsnr = (m_hat.pow(2) / (v_hat + 1e-8))
        return gsnr
    return torch.ones_like(p)

def train_epoch(model, loader, optimizer, criterion, mode, params):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if mode in ['GAD', 'GRD', 'NGRD']:
            p_base, gamma = params
            with torch.no_grad():
                if mode == 'GAD' or mode == 'GRD':
                    gsnr1 = get_gsnr_per_param(optimizer, model.fc1.weight).mean().item()
                    gsnr2 = get_gsnr_per_param(optimizer, model.fc2.weight).mean().item()
                    gsnr1 = max(0, min(1, gsnr1))
                    gsnr2 = max(0, min(1, gsnr2))

                    if mode == 'GAD':
                        p1 = p_base * ((1 - gsnr1) ** gamma)
                        p2 = p_base * ((1 - gsnr2) ** gamma)
                    else: # GRD
                        p1 = p_base * (gsnr1 ** gamma)
                        p2 = p_base * (gsnr2 ** gamma)

                    p1 = max(0, min(0.9, p1))
                    p2 = max(0, min(0.9, p2))
                    model.set_dropout_rates(p1, p2)

                elif mode == 'NGRD':
                    # Neuron-wise GSNR
                    gsnr1_w = get_gsnr_per_param(optimizer, model.fc1.weight) # [hidden, input]
                    gsnr2_w = get_gsnr_per_param(optimizer, model.fc2.weight) # [hidden, hidden]

                    gsnr1 = gsnr1_w.mean(dim=1) # [hidden]
                    gsnr2 = gsnr2_w.mean(dim=1) # [hidden]

                    gsnr1 = torch.clamp(gsnr1, 0, 1)
                    gsnr2 = torch.clamp(gsnr2, 0, 1)

                    p1 = p_base * (gsnr1 ** gamma)
                    p2 = p_base * (gsnr2 ** gamma)

                    p1 = torch.clamp(p1, 0, 0.9)
                    p2 = torch.clamp(p2, 0, 0.9)
                    model.set_neuron_dropout_rates(p1, p2)

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
        params = None
    elif mode == 'NGRD':
        p_base = trial.suggest_float('p_base', 0.0, 0.9)
        gamma = trial.suggest_float('gamma', 0.1, 5.0, log=True)
        model = NGRDMLP().to(device)
        params = (p_base, gamma)
    else: # GAD or GRD
        p_base = trial.suggest_float('p_base', 0.0, 0.9)
        gamma = trial.suggest_float('gamma', 0.1, 5.0, log=True)
        model = GRDMLP().to(device)
        params = (p_base, gamma)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(30):
        train_epoch(model, train_loader, optimizer, criterion, mode, params)
        _, val_acc = evaluate(model, val_loader, criterion)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def run_experiment():
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    data_for_obj = (x_train, y_train, x_val, y_val)

    modes = ['Baseline', 'GAD', 'GRD', 'NGRD']
    best_params = {}

    for mode in modes:
        print(f"Optimizing {mode}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode, data_for_obj), n_trials=20)
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
            p_params = None
        elif mode == 'NGRD':
            model = NGRDMLP().to(device)
            p_params = (params['p_base'], params['gamma'])
        else:
            model = GRDMLP().to(device)
            p_params = (params['p_base'], params['gamma'])

        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        criterion = nn.CrossEntropyLoss()

        history = {'train_loss': [], 'val_acc': []}
        best_val_acc = 0
        test_acc_at_best_val = 0

        for epoch in range(100):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, mode, p_params)
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
    plt.figure(figsize=(15, 5))
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
    plt.savefig('gsnr_regularized_dropout_experiment/results.png')

    # Save results to file
    with open('gsnr_regularized_dropout_experiment/results.txt', 'w') as f:
        for mode in modes:
            f.write(f"Mode: {mode}\n")
            f.write(f"Best Params: {results[mode]['best_params']}\n")
            f.write(f"Test Acc (at best val): {results[mode]['test_acc']:.4f}\n")
            f.write("-" * 20 + "\n")

if __name__ == '__main__':
    run_experiment()
