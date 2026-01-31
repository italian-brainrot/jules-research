import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import os
import matplotlib.pyplot as plt
from mnist1d.data import get_dataset_args, make_dataset

# Add current directory to path to import model
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import MLPClassifier, ZCAWhitening, SoftPCAWhitening, SpectralGatedWhitening

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

def get_data():
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    return X_train, y_train, X_test, y_test

def add_noise(x, noise_level=0.1):
    if noise_level <= 0:
        return x
    return x + noise_level * torch.randn_like(x)

def train_model(model, X_train, y_train, X_val, y_val, lr, weight_decay, train_noise=0.1, epochs=100):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0

    # We add noise to val data once
    X_val_noisy = add_noise(X_val, noise_level=train_noise)

    for epoch in range(epochs):
        model.train()
        # Reshuffle and add noise per batch
        indices = torch.randperm(X_train.shape[0])
        batch_size = 256
        for i in range(0, X_train.shape[0], batch_size):
            idx = indices[i:i+batch_size]
            batch_x = add_noise(X_train[idx], noise_level=train_noise)
            batch_y = y_train[idx]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_noisy)
                val_preds = val_outputs.argmax(dim=1)
                val_acc = (val_preds == y_val).float().mean().item()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

    return best_val_acc

def objective(trial, mode, X_train, y_train, X_val, y_val, train_noise):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256)

    preprocessing = None
    if mode == "zca":
        preprocessing = ZCAWhitening(X_train)
    elif mode == "soft_pca":
        eta = trial.suggest_float("eta", 1e-4, 1.0, log=True)
        preprocessing = SoftPCAWhitening(X_train, eta=eta)
    elif mode == "sgw":
        preprocessing = SpectralGatedWhitening(X_train)

    model = MLPClassifier(40, hidden_dim, 10, preprocessing=preprocessing)
    return train_model(model, X_train, y_train, X_val, y_val, lr, weight_decay, train_noise=train_noise, epochs=50)

def main():
    X_train_raw, y_train_raw, X_test, y_test = get_data()

    # Split train/val
    val_size = 1000
    X_val = X_train_raw[:val_size]
    y_val = y_train_raw[:val_size]
    X_train = X_train_raw[val_size:]
    y_train = y_train_raw[val_size:]

    train_noise = 0.2 # Add some noise during training to encourage learning filters
    test_noise_level = 0.5

    modes = ["none", "zca", "soft_pca", "sgw"]
    best_params = {}

    for mode in modes:
        print(f"\n--- Tuning mode: {mode} (Train Noise: {train_noise}) ---")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, mode, X_train, y_train, X_val, y_val, train_noise), n_trials=15)
        best_params[mode] = study.best_params
        print(f"Best params for {mode}: {best_params[mode]}")

    # Final evaluation
    results = {}
    X_test_clean = X_test
    X_test_noisy = add_noise(X_test, noise_level=test_noise_level)

    last_sgw_model = None

    for mode in modes:
        print(f"\n--- Final training for {mode} ---")
        params = best_params[mode]

        preprocessing = None
        if mode == "zca":
            preprocessing = ZCAWhitening(X_train)
        elif mode == "soft_pca":
            preprocessing = SoftPCAWhitening(X_train, eta=params['eta'])
        elif mode == "sgw":
            preprocessing = SpectralGatedWhitening(X_train)

        model = MLPClassifier(40, params['hidden_dim'], 10, preprocessing=preprocessing)
        train_model(model, X_train, y_train, X_val, y_val, params['lr'], params['weight_decay'], train_noise=train_noise, epochs=100)

        if mode == "sgw":
            last_sgw_model = model

        model.eval()
        with torch.no_grad():
            # Clean accuracy
            outputs = model(X_test_clean)
            acc_clean = (outputs.argmax(dim=1) == y_test).float().mean().item()

            # Noisy accuracy
            outputs_noisy = model(X_test_noisy)
            acc_noisy = (outputs_noisy.argmax(dim=1) == y_test).float().mean().item()

            results[mode] = {"Clean": acc_clean, "Noisy": acc_noisy}
            print(f"Mode {mode}: Clean Acc = {acc_clean:.4f}, Noisy Acc = {acc_noisy:.4f}")
            if mode == "sgw":
                log_eigvals = torch.log(model.preprocessing.eigvals + 1e-8).unsqueeze(1)
                s = model.preprocessing.gate(log_eigvals).squeeze(1)
                print("SGW Gate values (first 5, last 5):", s[:5].detach().cpu().numpy(), s[-5:].detach().cpu().numpy())

    # Plot results
    labels = list(results.keys())
    clean_accs = [results[m]['Clean'] for m in labels]
    noisy_accs = [results[m]['Noisy'] for m in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, clean_accs, width, label='Clean Test Acc')
    ax.bar(x + width/2, noisy_accs, width, label=f'Noisy Test Acc (std={test_noise_level})')

    ax.set_ylabel('Accuracy')
    ax.set_title(f'Comparison of Whitening Methods (Train Noise={train_noise})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig("spectral_gated_whitening_experiment/results.png")

    # Plot SGW learned gate
    if last_sgw_model is not None:
        last_sgw_model.eval()
        with torch.no_grad():
            eigvals = last_sgw_model.preprocessing.eigvals
            log_eigvals = torch.log(eigvals + 1e-8).unsqueeze(1)
            s = last_sgw_model.preprocessing.gate(log_eigvals).squeeze(1).cpu().numpy()
            eigvals_np = eigvals.cpu().numpy()

            plt.figure(figsize=(10, 6))
            plt.semilogx(eigvals_np, s, 'o-')
            plt.xlabel('Eigenvalue (log scale)')
            plt.ylabel('Gate value (s)')
            plt.title('Learned Spectral Gate for SGW (Trained with Noise)')
            plt.grid(True)
            plt.savefig("spectral_gated_whitening_experiment/sgw_gate.png")

    # Save results to text
    with open("spectral_gated_whitening_experiment/results.txt", "w") as f:
        f.write("Mode,Clean Accuracy,Noisy Accuracy\n")
        for mode in labels:
            f.write(f"{mode},{results[mode]['Clean']:.4f},{results[mode]['Noisy']:.4f}\n")

if __name__ == "__main__":
    main()
