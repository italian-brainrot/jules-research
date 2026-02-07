import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torch.func import vmap, grad, functional_call
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader

# Add current directory to sys.path to import model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import MLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 5000 # Increased for better statistics
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def get_sgsc_grads(model, x, y, gamma):
    params = dict(model.named_parameters())
    names = params.keys()
    values = tuple(params.values())

    def compute_loss(params_values, x_single, y_single):
        p_dict = {name: val for name, val in zip(names, params_values)}
        out = functional_call(model, p_dict, (x_single.unsqueeze(0),))
        loss = F.cross_entropy(out, y_single.unsqueeze(0))
        return loss

    grad_fn = grad(compute_loss)
    v_grad_fn = vmap(grad_fn, in_dims=(None, 0, 0))

    per_sample_grads = v_grad_fn(values, x, y)

    new_grads = []
    for ps_grad in per_sample_grads:
        avg_grad = ps_grad.mean(dim=0)
        signs = torch.sign(ps_grad)
        consistency = torch.abs(signs.mean(dim=0))
        new_grad = avg_grad * (consistency ** gamma)
        new_grads.append(new_grad)

    return new_grads

def train_model(X_train, y_train, X_test, y_test, use_sgsc=False, lr=1e-3, gamma=1.0, epochs=20, batch_size=128, device='cpu'):
    model = MLP(40, [256, 256], 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = TensorDataLoader((X_train.to(device), y_train.to(device)), batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            if use_sgsc:
                new_grads = get_sgsc_grads(model, batch_X, batch_y, gamma)
                for p, g in zip(model.parameters(), new_grads):
                    p.grad = g
                # We need a loss value for tracking, but we already have grads
                with torch.no_grad():
                    out = model(batch_X)
                    loss = F.cross_entropy(out, batch_y)
            else:
                out = model(batch_X)
                loss = F.cross_entropy(out, batch_y)
                loss.backward()

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            out = model(X_test.to(device))
            acc = (out.argmax(1) == y_test.to(device)).float().mean().item()

        history["train_loss"].append(avg_loss)
        history["test_acc"].append(acc)

    return model, history

def objective(trial, X_train, y_train, X_test, y_test, use_sgsc, device):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    gamma = 0.0
    if use_sgsc:
        gamma = trial.suggest_float("gamma", 0.0, 3.0)

    _, history = train_model(X_train, y_train, X_test, y_test, use_sgsc=use_sgsc, lr=lr, gamma=gamma, epochs=10, device=device)
    return max(history["test_acc"])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test = get_data()

    print("Tuning Adam baseline...")
    study_adam = optuna.create_study(direction="maximize")
    study_adam.optimize(lambda t: objective(t, X_train, y_train, X_test, y_test, False, device), n_trials=15)
    best_lr_adam = study_adam.best_params["lr"]

    print("Tuning Adam + SGSC...")
    study_sgsc = optuna.create_study(direction="maximize")
    study_sgsc.optimize(lambda t: objective(t, X_train, y_train, X_test, y_test, True, device), n_trials=20)
    best_lr_sgsc = study_sgsc.best_params["lr"]
    best_gamma = study_sgsc.best_params["gamma"]

    print(f"Best Adam LR: {best_lr_adam}")
    print(f"Best SGSC LR: {best_lr_sgsc}, Gamma: {best_gamma}")

    # Final evaluation
    seeds = [42, 43, 44]
    results = {"adam": [], "sgsc": []}

    for seed in seeds:
        torch.manual_seed(seed)
        _, hist_adam = train_model(X_train, y_train, X_test, y_test, use_sgsc=False, lr=best_lr_adam, epochs=50, device=device)
        results["adam"].append(hist_adam)

        torch.manual_seed(seed)
        _, hist_sgsc = train_model(X_train, y_train, X_test, y_test, use_sgsc=True, lr=best_lr_sgsc, gamma=best_gamma, epochs=50, device=device)
        results["sgsc"].append(hist_sgsc)

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for label, res_list in results.items():
        accs = np.array([h["test_acc"] for h in res_list])
        plt.plot(accs.mean(0), label=label)
        plt.fill_between(range(50), accs.min(0), accs.max(0), alpha=0.2)
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    for label, res_list in results.items():
        losses = np.array([h["train_loss"] for h in res_list])
        plt.plot(losses.mean(0), label=label)
        plt.fill_between(range(50), losses.min(0), losses.max(0), alpha=0.2)
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("gradient_sign_consistency_experiment/comparison.png")

    with open("gradient_sign_consistency_experiment/results.txt", "w") as f:
        f.write(f"Adam: {np.mean([h['test_acc'][-1] for h in results['adam']]):.4f}\n")
        f.write(f"SGSC: {np.mean([h['test_acc'][-1] for h in results['sgsc']]):.4f}\n")
        f.write(f"Best SGSC Gamma: {best_gamma:.4f}\n")
        f.write(f"Best SGSC LR: {best_lr_sgsc:.6f}\n")
        f.write(f"Best Adam LR: {best_lr_adam:.6f}\n")

if __name__ == "__main__":
    main()
