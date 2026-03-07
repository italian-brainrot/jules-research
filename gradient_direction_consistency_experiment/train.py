import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from torch.func import vmap, grad, functional_call
import os

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

def compute_gdcr_loss(model, params, x, y):
    def compute_loss(params, x_single, y_single):
        logits = functional_call(model, params, (x_single.unsqueeze(0),))
        return F.cross_entropy(logits, y_single.unsqueeze(0))

    # Compute per-sample gradients w.r.t. all parameters
    per_sample_grads = vmap(grad(compute_loss), in_dims=(None, 0, 0))(params, x, y)

    # Flatten and concatenate all gradients for each sample
    flat_grads = []
    for p_name in params:
        g = per_sample_grads[p_name]
        flat_grads.append(g.reshape(x.shape[0], -1))

    all_flat_grads = torch.cat(flat_grads, dim=1) # (batch_size, num_params)

    # Normalize gradients to get directions
    norms = torch.norm(all_flat_grads, p=2, dim=1, keepdim=True) + 1e-8
    grad_directions = all_flat_grads / norms

    # Compute average cosine similarity efficiently
    batch_size = x.shape[0]
    sum_grads = grad_directions.sum(dim=0)
    sum_sq_norm = torch.sum(sum_grads**2)
    avg_cos_sim = (sum_sq_norm - batch_size) / (batch_size * (batch_size - 1))

    # GDCR loss: minimize (1 - avg_cos_sim)
    gdcr_loss = 1.0 - avg_cos_sim

    return gdcr_loss

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

def train_epoch(model, loader, optimizer, mode, lambda_gdcr, device):
    model.train()
    total_loss = 0
    params = dict(model.named_parameters())
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        ce_loss = F.cross_entropy(logits, y)

        if mode == 'GDCR':
            gdcr_loss = compute_gdcr_loss(model, params, x, y)
            loss = ce_loss + lambda_gdcr * gdcr_loss
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

    lambda_gdcr = 0.0
    if mode == 'GDCR':
        lambda_gdcr = trial.suggest_float('lambda_gdcr', 1e-3, 1.0, log=True)

    model = MLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(10):
        train_epoch(model, train_loader, optimizer, mode, lambda_gdcr, device)

    val_acc = evaluate(model, val_loader, device)
    return val_acc

if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    data = (x_train, y_train, x_val, y_val)

    for mode in ['Baseline', 'GDCR']:
        # Check if already tuned
        if os.path.exists(f'gradient_direction_consistency_experiment/best_params_{mode}.txt'):
            print(f"Skipping tuning for {mode}, already exists.")
            continue

        print(f"Tuning {mode}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode, data), n_trials=10)
        print(f"Best {mode} params: {study.best_params}")

        # Save best params
        with open(f'gradient_direction_consistency_experiment/best_params_{mode}.txt', 'w') as f:
            f.write(str(study.best_params))
