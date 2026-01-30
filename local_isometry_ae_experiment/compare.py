import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import Autoencoder, cae_penalty, lip_penalty, LatentClassifier
import os

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# Load data
print("Loading MNIST1D data...")
defaults = get_dataset_args()
defaults.num_samples = 5000
data = make_dataset(defaults)

X_train = torch.tensor(data['x']).float()
y_train = torch.tensor(data['y']).long()
X_test = torch.tensor(data['x_test']).float()
y_test = torch.tensor(data['y_test']).long()

LATENT_DIM = 2

def evaluate_latent_accuracy(ae, X_train_val, y_train_val, X_test_val, y_test_val):
    ae.eval()
    with torch.no_grad():
        Z_train = ae.encoder(X_train_val)
        Z_test = ae.encoder(X_test_val)

    classifier = LatentClassifier(latent_dim=LATENT_DIM)
    optimizer = optim.Adam(classifier.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train classifier
    for _ in range(100):
        optimizer.zero_grad()
        output = classifier(Z_train)
        loss = criterion(output, y_train_val)
        loss.backward()
        optimizer.step()

    # Evaluate
    classifier.eval()
    with torch.no_grad():
        output = classifier(Z_test)
        pred = output.argmax(dim=1)
        acc = (pred == y_test_val).float().mean().item()

    return acc

def train_ae(model_type, lr, lmbda=0, epochs=30):
    ae = Autoencoder(latent_dim=LATENT_DIM)
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

    for epoch in range(epochs):
        ae.train()
        for x_batch, _ in dl_train:
            optimizer.zero_grad()
            x_hat, z = ae(x_batch)
            loss_rec = criterion(x_hat, x_batch)

            if model_type == 'CAE':
                penalty = cae_penalty(ae.encoder, x_batch)
                loss = loss_rec + lmbda * penalty
            elif model_type == 'LIP':
                penalty = lip_penalty(ae.encoder, x_batch)
                loss = loss_rec + lmbda * penalty
            else:
                loss = loss_rec

            loss.backward()
            optimizer.step()

    return ae

def objective(trial, model_type):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lmbda = 0
    if model_type in ['CAE', 'LIP']:
        lmbda = trial.suggest_float("lmbda", 1e-4, 1e-1, log=True)

    ae = train_ae(model_type, lr, lmbda, epochs=20)
    acc = evaluate_latent_accuracy(ae, X_train, y_train, X_test, y_test)
    return acc

def run_experiment():
    results = {}

    for model_type in ['Baseline', 'CAE', 'LIP']:
        print(f"--- Tuning {model_type} ---")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, model_type), n_trials=15)

        best_lr = study.best_params['lr']
        best_lmbda = study.best_params.get('lmbda', 0)
        print(f"Best {model_type}: LR={best_lr:.6f}, Lambda={best_lmbda:.6f}, Acc={study.best_value:.4f}")

        print(f"Final training for {model_type}...")
        ae = train_ae(model_type, best_lr, best_lmbda, epochs=50)
        acc = evaluate_latent_accuracy(ae, X_train, y_train, X_test, y_test)

        ae.eval()
        with torch.no_grad():
            x_hat_test, _ = ae(X_test)
            mse = nn.MSELoss()(x_hat_test, X_test).item()
            Z_test = ae.encoder(X_test).numpy()

        results[model_type] = {
            'acc': acc,
            'mse': mse,
            'Z': Z_test,
            'params': study.best_params
        }

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, model_type in enumerate(['Baseline', 'CAE', 'LIP']):
        Z = results[model_type]['Z']
        sc = axes[i].scatter(Z[:, 0], Z[:, 1], c=y_test.numpy(), cmap='tab10', s=2, alpha=0.5)
        axes[i].set_title(f"{model_type}\nAcc: {results[model_type]['acc']:.4f}, MSE: {results[model_type]['mse']:.4f}")
        if i == 2:
            plt.colorbar(sc, ax=axes[i])

    plt.tight_layout()
    plt.savefig("local_isometry_ae_experiment/latent_space.png")

    # Print summary table
    print("\nSummary Results:")
    print(f"{'Model':<10} | {'MSE':<10} | {'Accuracy':<10}")
    print("-" * 35)
    for model_type, res in results.items():
        print(f"{model_type:<10} | {res['mse']:<10.6f} | {res['acc']:<10.4f}")

    # Write to README.md
    with open("local_isometry_ae_experiment/results.md", "w") as f:
        f.write("# Experiment Results\n\n")
        f.write("| Model | MSE | Accuracy | Best Params |\n")
        f.write("|-------|-----|----------|-------------|\n")
        for model_type, res in results.items():
            f.write(f"| {model_type} | {res['mse']:.6f} | {res['acc']:.4f} | {res['params']} |\n")

if __name__ == "__main__":
    run_experiment()
