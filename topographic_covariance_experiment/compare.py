import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data():
    if hasattr(get_data, "cache"):
        return get_data.cache
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X = torch.tensor(data['x'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    # Split X, y into train and val
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    get_data.cache = (X_train, y_train, X_val, y_val, X_test, y_test)
    return get_data.cache

class MLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_embedding=False):
        z1 = F.relu(self.fc1(x))
        z2 = F.relu(self.fc2(z1))
        logits = self.classifier(z2)
        if return_embedding:
            return logits, z2
        return logits

def get_target_covariance(dim, sigma, device):
    if sigma <= 0:
        return torch.eye(dim, device=device)

    indices = torch.arange(dim, device=device)
    # Circular distance on a 1D ring
    dist = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
    dist = torch.min(dist, dim - dist)

    target = torch.exp(-(dist.float()**2) / (2 * sigma**2))
    return target

def tcr_loss(embeddings, target_cov):
    # Center the embeddings
    z_mean = embeddings.mean(dim=0, keepdim=True)
    z_centered = embeddings - z_mean

    batch_size = embeddings.size(0)
    if batch_size <= 1:
        return torch.tensor(0.0, device=embeddings.device)

    # Compute covariance
    cov = (z_centered.t() @ z_centered) / (batch_size - 1)

    # MSE between cov and target_cov
    loss = F.mse_loss(cov, target_cov)
    return loss

def train_eval(lr, epochs, method="ce", method_params=None, is_final=False):
    X_train, y_train, X_val, y_val, X_test, y_test = get_data()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)

    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dim = 256
    if method != "ce":
        sigma = method_params.get("sigma", 0.0)
        lambda_reg = method_params.get("lambda", 0.1)
        target_cov = get_target_covariance(dim, sigma, device)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in dl_train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if method != "ce":
                logits, z = model(inputs, return_embedding=True)
                loss_ce = F.cross_entropy(logits, targets)
                loss_reg = tcr_loss(z, target_cov)
                loss = loss_ce + lambda_reg * loss_reg
            else:
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)

            loss.backward()
            optimizer.step()

    model.eval()
    if is_final:
        eval_X, eval_y = X_test, y_test
    else:
        eval_X, eval_y = X_val, y_val

    with torch.no_grad():
        test_logits = model(eval_X.to(device))
        preds = test_logits.argmax(dim=1)
        acc = (preds == eval_y.to(device)).float().mean().item()

    if is_final:
        return acc, model
    return acc

def test_robustness(model, X_test, y_test, dropout_type="random", dropout_rate=0.2):
    model.eval()
    dim = 256

    with torch.no_grad():
        # Get embeddings
        z1 = F.relu(model.fc1(X_test.to(device)))
        z2 = F.relu(model.fc2(z1))

        if dropout_type == "random":
            mask = (torch.rand(z2.shape, device=device) > dropout_rate).float()
        elif dropout_type == "contiguous":
            # Drop a contiguous block of features for each sample
            mask = torch.ones_like(z2)
            block_size = int(dim * dropout_rate)
            for i in range(z2.size(0)):
                start = torch.randint(0, dim, (1,)).item()
                indices = (torch.arange(start, start + block_size)) % dim
                mask[i, indices] = 0.0

        z2_dropped = z2 * mask
        logits = model.classifier(z2_dropped)
        preds = logits.argmax(dim=1)
        acc = (preds == y_test.to(device)).float().mean().item()
    return acc

def plot_covariance(model, X_test, title, filename):
    model.eval()
    with torch.no_grad():
        _, z = model(X_test.to(device), return_embedding=True)
        z_mean = z.mean(dim=0, keepdim=True)
        z_centered = z - z_mean
        cov = (z_centered.t() @ z_centered) / (z.size(0) - 1)
        cov = cov.cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(cov, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def objective(trial, method):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    params = {}
    if method == "whitening":
        params["lambda"] = trial.suggest_float("lambda", 1e-3, 100.0, log=True)
        params["sigma"] = 0.0
    elif method == "tcr":
        params["lambda"] = trial.suggest_float("lambda", 1e-3, 100.0, log=True)
        params["sigma"] = trial.suggest_float("sigma", 0.5, 20.0)

    epochs = 40
    return train_eval(lr, epochs, method, params, is_final=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    n_trials = 15
    epochs = 80
    if args.smoke_test:
        n_trials = 1
        epochs = 1

    results = {}
    X_train, y_train, X_val, y_val, X_test, y_test = get_data()

    methods = ["ce", "whitening", "tcr"]
    for method in methods:
        print(f"Tuning {method}...")
        study = optuna.create_study(direction="maximize")
        if args.smoke_test:
            study.optimize(lambda t: train_eval(1e-3, 1, method, {"lambda": 1.0, "sigma": 2.0}, is_final=False), n_trials=n_trials)
        else:
            study.optimize(lambda t: objective(t, method), n_trials=n_trials)

        best_params = study.best_params
        if args.smoke_test:
            best_params = {"lr": 1e-3}
            if method == "whitening": best_params["lambda"] = 1.0; best_params["sigma"] = 0.0
            if method == "tcr": best_params["lambda"] = 1.0; best_params["sigma"] = 2.0
        elif method == "ce":
            best_params = {"lr": study.best_params["lr"]}
        elif method == "whitening":
            best_params["sigma"] = 0.0

        print(f"Training final {method} with best params: {best_params}")
        final_acc, model = train_eval(best_params["lr"], epochs, method, best_params, is_final=True)

        # Robustness tests
        acc_random_drop = test_robustness(model, X_test, y_test, dropout_type="random", dropout_rate=0.2)
        acc_contiguous_drop = test_robustness(model, X_test, y_test, dropout_type="contiguous", dropout_rate=0.2)

        results[method] = {
            "best_params": best_params,
            "final_acc": final_acc,
            "acc_random_drop": acc_random_drop,
            "acc_contiguous_drop": acc_contiguous_drop
        }

        # Plot covariance
        plot_covariance(model, X_test, f"Covariance: {method}", f"topographic_covariance_experiment/cov_{method}.png")

    # Save results
    with open("topographic_covariance_experiment/results.txt", "w") as f:
        for method, res in results.items():
            f.write(f"Method: {method}\n")
            f.write(f"Best Params: {res['best_params']}\n")
            f.write(f"Final Test Accuracy: {res['final_acc']:.4f}\n")
            f.write(f"Accuracy (Random Dropout 20%): {res['acc_random_drop']:.4f}\n")
            f.write(f"Accuracy (Contiguous Dropout 20%): {res['acc_contiguous_drop']:.4f}\n")
            f.write("-" * 20 + "\n")

    print("\nFinal Results:")
    for method, res in results.items():
        print(f"{method}: {res['final_acc']:.4f} (Random Drop: {res['acc_random_drop']:.4f}, Contiguous Drop: {res['acc_contiguous_drop']:.4f})")
