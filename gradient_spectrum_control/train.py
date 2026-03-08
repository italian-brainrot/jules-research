import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from torch.func import functional_call, vmap, grad
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, output_size=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data():
    args = get_dataset_args()
    args.num_samples = 2000
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

def compute_loss(params, model, x, y):
    logits = functional_call(model, params, (x.unsqueeze(0),))
    return nn.functional.cross_entropy(logits, y.unsqueeze(0))

def get_spectral_ratio(params, model, x_batch, y_batch):
    batch_size = x_batch.shape[0]

    def single_loss(p, x, y):
        return compute_loss(p, model, x, y)

    per_sample_grads_dict = vmap(grad(single_loss), in_dims=(None, 0, 0))(params, x_batch, y_batch)

    flat_grads = []
    for p in per_sample_grads_dict.values():
        flat_grads.append(p.reshape(batch_size, -1))
    G = torch.cat(flat_grads, dim=1)  # (B, P)

    G_norm = torch.norm(G, dim=1, keepdim=True) + 1e-8
    G_normalized = G / G_norm

    K = torch.mm(G_normalized, G_normalized.t())
    L = torch.linalg.eigvalsh(K)
    L = torch.relu(L)

    max_eig = L[-1]
    total_eig = L.sum()
    ratio = max_eig / (total_eig + 1e-8)
    return ratio

def train_epoch(model, loader, optimizer, mode, lambda_reg=0.0):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Standard loss
        output = model(x)
        ce_loss = nn.functional.cross_entropy(output, y)

        if mode != 'Baseline':
            params = dict(model.named_parameters())
            ratio = get_spectral_ratio(params, model, x, y)
            if mode == 'GSCR_Consensus':
                # Minimize -ratio to maximize consensus
                penalty = -ratio
            elif mode == 'GSCR_Diversity':
                # Minimize ratio to maximize diversity
                penalty = ratio
            else:
                penalty = torch.tensor(0.0, device=device)

            loss = ce_loss + lambda_reg * penalty
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
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    if mode != 'Baseline':
        lambda_reg = trial.suggest_float('lambda_reg', 1e-3, 1.0, log=True)
    else:
        lambda_reg = 0.0

    model = MLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_val_acc = 0
    for epoch in range(10):
        train_epoch(model, train_loader, optimizer, mode, lambda_reg)
        val_acc = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    return best_val_acc

def run_experiment():
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    data_for_obj = (x_train, y_train, x_val, y_val)

    modes = ['Baseline', 'GSCR_Consensus', 'GSCR_Diversity']
    best_params = {}

    for mode in modes:
        print(f"Optimizing {mode}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode, data_for_obj), n_trials=5)
        best_params[mode] = study.best_params
        print(f"Best params for {mode}: {study.best_params}")

    # Final training
    train_loader = TensorDataLoader((x_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((x_val, y_val), batch_size=128, shuffle=False)
    test_loader = TensorDataLoader((x_test, y_test), batch_size=128, shuffle=False)

    histories = {}
    seeds = [42, 43, 44]

    for mode in modes:
        print(f"Final training for {mode}...")
        params = best_params[mode]
        mode_accs = []

        # We'll plot the first seed
        history = {'train_loss': [], 'val_acc': [], 'test_acc': []}

        for i, seed in enumerate(seeds):
            set_seed(seed)
            model = MLP().to(device)
            optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            lambda_reg = params.get('lambda_reg', 0.0)

            best_v_acc = 0
            t_acc_at_best_v = 0

            for epoch in range(20):
                loss = train_epoch(model, train_loader, optimizer, mode, lambda_reg)
                v_acc = evaluate(model, val_loader)
                t_acc = evaluate(model, test_loader)

                if v_acc > best_v_acc:
                    best_v_acc = v_acc
                    t_acc_at_best_v = t_acc

                if i == 0:
                    history['train_loss'].append(loss)
                    history['val_acc'].append(v_acc)
                    history['test_acc'].append(t_acc)

            mode_accs.append(t_acc_at_best_v)
            print(f"Mode {mode}, Seed {seed}, Test Acc: {t_acc_at_best_v:.4f}")

        histories[mode] = {
            'accs': mode_accs,
            'mean': np.mean(mode_accs),
            'std': np.std(mode_accs),
            'history': history,
            'params': params
        }

    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for mode in modes:
        plt.plot(histories[mode]['history']['train_loss'], label=mode)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    for mode in modes:
        plt.plot(histories[mode]['history']['test_acc'], label=mode)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('gradient_spectrum_control/results.png')

    # Save summary
    with open('gradient_spectrum_control/results_summary.txt', 'w') as f:
        for mode in modes:
            res = histories[mode]
            f.write(f"Mode: {mode}\n")
            f.write(f"Mean Test Acc: {res['mean']:.4f} ± {res['std']:.4f}\n")
            f.write(f"Best Params: {res['params']}\n")
            f.write("-" * 20 + "\n")

    print("Experiment completed.")

if __name__ == "__main__":
    run_experiment()
