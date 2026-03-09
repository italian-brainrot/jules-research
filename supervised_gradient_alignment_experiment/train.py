import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from torch.func import vmap, grad, functional_call
import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class MLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, output_size=10):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

def compute_sga_loss(model, params, x, y):
    def compute_loss(params, x_single, y_single):
        logits = functional_call(model, params, (x_single.unsqueeze(0),))
        return F.cross_entropy(logits, y_single.unsqueeze(0))

    per_sample_grads = vmap(grad(compute_loss), in_dims=(None, 0, 0))(params, x, y)

    flat_grads = []
    for p_name in params:
        g = per_sample_grads[p_name]
        flat_grads.append(g.reshape(x.shape[0], -1))

    all_flat_grads = torch.cat(flat_grads, dim=1)

    norms = torch.norm(all_flat_grads, p=2, dim=1, keepdim=True) + 1e-8
    grad_directions = all_flat_grads / norms

    sim_matrix = torch.matmul(grad_directions, grad_directions.t())

    batch_size = x.shape[0]
    y_vec = y.unsqueeze(0)
    mask_intra = (y_vec == y_vec.t()).float()
    mask_inter = 1.0 - mask_intra
    mask_intra = mask_intra - torch.eye(batch_size, device=y.device)

    intra_count = mask_intra.sum()
    intra_sim = (sim_matrix * mask_intra).sum() / intra_count if intra_count > 0 else torch.tensor(0.0, device=y.device)

    inter_count = mask_inter.sum()
    inter_sim = (sim_matrix * mask_inter).sum() / inter_count if inter_count > 0 else torch.tensor(0.0, device=y.device)

    return intra_sim, inter_sim

def get_data():
    args = get_dataset_args()
    args.num_samples = 4000
    data = make_dataset(args)
    x_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    x_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    n_train = int(0.8 * len(x_train))
    x_val = x_train[n_train:]
    y_val = y_train[n_train:]
    x_train = x_train[:n_train]
    y_train = y_train[:n_train]

    return x_train, y_train, x_val, y_val, x_test, y_test

def train_epoch(model, loader, optimizer, mode, lambda_intra, lambda_inter, device):
    model.train()
    total_loss = 0
    params = dict(model.named_parameters())
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        ce_loss = F.cross_entropy(logits, y)

        if mode == 'SGA':
            intra_sim, inter_sim = compute_sga_loss(model, params, x, y)
            # We want to maximize intra_sim and minimize inter_sim
            # So minimize (1 - intra_sim) and maximize (-inter_sim) -> minimize inter_sim
            loss = ce_loss + lambda_intra * (1.0 - intra_sim) + lambda_inter * inter_sim
        else:
            loss = ce_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
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

    lambda_intra = 0.0
    lambda_inter = 0.0
    if mode == 'SGA':
        lambda_intra = trial.suggest_float('lambda_intra', 1e-3, 1.0, log=True)
        lambda_inter = trial.suggest_float('lambda_inter', 1e-3, 1.0, log=True)

    model = MLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(10):
        train_epoch(model, train_loader, optimizer, mode, lambda_intra, lambda_inter, device)

    val_acc = evaluate(model, val_loader, device)
    return val_acc

if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    data = (x_train, y_train, x_val, y_val)

    results_dir = 'supervised_gradient_alignment_experiment'
    os.makedirs(results_dir, exist_ok=True)

    for mode in ['Baseline', 'SGA']:
        params_path = os.path.join(results_dir, f'best_params_{mode}.json')
        if os.path.exists(params_path):
            print(f"Skipping tuning for {mode}, already exists.")
            continue

        print(f"Tuning {mode}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode, data), n_trials=10)
        print(f"Best {mode} params: {study.best_params}")

        with open(params_path, 'w') as f:
            json.dump(study.best_params, f)
