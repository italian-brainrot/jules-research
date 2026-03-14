import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import vmap, grad, functional_call
import numpy as np
import optuna
import os
import argparse
import json
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader

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

def augment_data(x, shift_range=5, noise_std=0.01):
    batch_size, seq_len = x.shape
    shifts = torch.randint(-shift_range, shift_range + 1, (batch_size,), device=x.device)
    # Roll is a bit slow in a loop, but for small batch and 1D it's fine.
    # We use a list comprehension and stack for simplicity here.
    x_aug = torch.stack([torch.roll(x[i], shifts[i].item(), dims=0) for i in range(batch_size)])
    noise = torch.randn_like(x_aug) * noise_std
    x_aug = x_aug + noise
    return x_aug

def compute_loss_single(params, buffers, model, x_single, y_single):
    logits = functional_call(model, (params, buffers), (x_single.unsqueeze(0),))
    return F.cross_entropy(logits, y_single.unsqueeze(0))

def get_data(num_samples=4000):
    args = get_dataset_args()
    args.num_samples = num_samples
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

def train_epoch(model, loader, optimizer, lambda_aiga=0.0, use_aug=False):
    model.train()
    total_loss = 0
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    grad_fn = vmap(grad(compute_loss_single), in_dims=(None, None, None, 0, 0))

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Data for CE loss (can be augmented or original)
        if use_aug:
            x_ce = augment_data(x)
        else:
            x_ce = x

        logits = model(x_ce)
        ce_loss = F.cross_entropy(logits, y)

        if lambda_aiga > 0:
            # Per-sample gradients for original and augmented
            grads_orig = grad_fn(params, buffers, model, x, y)
            x_aug = augment_data(x)
            grads_aug = grad_fn(params, buffers, model, x_aug, y)

            def flatten_grads(grads_dict):
                flat_grads = []
                for name in sorted(grads_dict.keys()):
                    g = grads_dict[name]
                    flat_grads.append(g.reshape(x.shape[0], -1))
                return torch.cat(flat_grads, dim=1)

            f_orig = flatten_grads(grads_orig)
            f_aug = flatten_grads(grads_aug)

            norm_orig = torch.norm(f_orig, p=2, dim=1, keepdim=True) + 1e-8
            norm_aug = torch.norm(f_aug, p=2, dim=1, keepdim=True) + 1e-8

            cos_sim = torch.sum((f_orig / norm_orig) * (f_aug / norm_aug), dim=1)
            aiga_loss = 1.0 - torch.mean(cos_sim)

            loss = ce_loss + lambda_aiga * aiga_loss
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

    lambda_aiga = 0.0
    use_aug = True # Both baseline and AIGA use augmentations for CE loss for fair comparison

    if mode == 'AIGA':
        lambda_aiga = trial.suggest_float('lambda_aiga', 0.01, 10.0, log=True)

    model = MLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_val_acc = 0
    num_epochs = 15 if mode == 'Baseline' else 10
    for epoch in range(num_epochs):
        train_epoch(model, train_loader, optimizer, lambda_aiga, use_aug)
        val_acc = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='AIGA', choices=['AIGA', 'Baseline'])
    args_opt = parser.parse_args()

    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    data = (x_train, y_train, x_val, y_val)

    study_name = f"aiga_{args_opt.mode}"
    study = optuna.create_study(direction='maximize', study_name=study_name)
    n_trials = 20 if args_opt.mode == 'Baseline' else 5
    study.optimize(lambda trial: objective(trial, args_opt.mode, data), n_trials=n_trials)

    print(f"Best params for {args_opt.mode}: {study.best_params}")

    os.makedirs('augmentation_invariant_gradient_alignment', exist_ok=True)
    with open(f'augmentation_invariant_gradient_alignment/best_params_{args_opt.mode}.json', 'w') as f:
        json.dump(study.best_params, f)
