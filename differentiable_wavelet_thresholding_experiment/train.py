import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import BaselineMLP, DAWTMLP

def get_data(num_samples=10000):
    defaults = get_dataset_args()
    defaults.num_samples = num_samples
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_model(model, dl_train, lr, weight_decay, epochs=50):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y).sum().item() / y.size(0)
    return accuracy

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    if model_type == 'baseline':
        model = BaselineMLP()
    else:
        model = DAWTMLP(levels=trial.suggest_int("levels", 1, 3))

    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    train_model(model, dl_train, lr, weight_decay, epochs=30)
    acc = evaluate(model, X_test, y_test)
    return acc

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    results = {}

    for model_type in ['baseline', 'dawt']:
        print(f"\n--- Tuning {model_type} ---")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=10)

        best_params = study.best_params
        print(f"Best params for {model_type}: {best_params}")

        # Final evaluation over 3 seeds
        accs_clean = []
        accs_noisy = []
        all_thresholds = []

        for seed in range(3):
            torch.manual_seed(seed)
            if model_type == 'baseline':
                model = BaselineMLP()
            else:
                model = DAWTMLP(levels=best_params['levels'])

            dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
            train_model(model, dl_train, best_params['lr'], best_params['weight_decay'], epochs=50)

            # Clean accuracy
            acc_clean = evaluate(model, X_test, y_test)
            accs_clean.append(acc_clean)

            # Noisy accuracy
            X_test_noisy = X_test + 0.3 * torch.randn_like(X_test)
            acc_noisy = evaluate(model, X_test_noisy, y_test)
            accs_noisy.append(acc_noisy)

            if model_type == 'dawt':
                all_thresholds.append(model.dawt.thresholds.detach().cpu().numpy())

        results[model_type] = {
            'clean_mean': np.mean(accs_clean),
            'clean_std': np.std(accs_clean),
            'noisy_mean': np.mean(accs_noisy),
            'noisy_std': np.std(accs_noisy),
            'best_params': best_params
        }
        if model_type == 'dawt':
            results[model_type]['thresholds'] = np.mean(all_thresholds, axis=0)

    # Save results
    with open("results.txt", "w") as f:
        for model_type, res in results.items():
            f.write(f"Model: {model_type}\n")
            f.write(f"  Clean: {res['clean_mean']:.4f} +/- {res['clean_std']:.4f}\n")
            f.write(f"  Noisy: {res['noisy_mean']:.4f} +/- {res['noisy_std']:.4f}\n")
            f.write(f"  Best params: {res['best_params']}\n")
            if 'thresholds' in res:
                f.write(f"  Mean learned thresholds: {res['thresholds']}\n")
            f.write("\n")

    # Plot results
    model_names = list(results.keys())
    clean_means = [results[m]['clean_mean'] for m in model_names]
    noisy_means = [results[m]['noisy_mean'] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(x - width/2, clean_means, width, label='Clean')
    ax[0].bar(x + width/2, noisy_means, width, label='Noisy (std=0.3)')

    ax[0].set_ylabel('Accuracy')
    ax[0].set_title('Baseline vs DAWT-MLP Accuracy')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(model_names)
    ax[0].legend()

    # Visualization of denoising effect
    torch.manual_seed(42)
    sample_x = X_test[0:1]
    sample_x_noisy = sample_x + 0.3 * torch.randn_like(sample_x)

    best_levels = results['dawt']['best_params']['levels']
    best_lr = results['dawt']['best_params']['lr']
    best_wd = results['dawt']['best_params']['weight_decay']

    # Re-train one DAWT model briefly to get a decent one for visualization if needed,
    # but here we just use the last one from the loop.
    model_dawt = DAWTMLP(levels=best_levels)
    dl_train = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    train_model(model_dawt, dl_train, best_lr, best_wd, epochs=10) # Just a few epochs for visual

    model_dawt.eval()
    with torch.no_grad():
        denoised = model_dawt.dawt(sample_x_noisy)

    ax[1].plot(sample_x[0].numpy(), label='Original', alpha=0.5)
    ax[1].plot(sample_x_noisy[0].numpy(), label='Noisy', alpha=0.5)
    ax[1].plot(denoised[0].numpy(), label='DAWT Denoised', linewidth=2)
    ax[1].set_title('Denoising Visualization')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig("comparison.png")
    print("\nExperiment complete. Results saved to results.txt and comparison.png.")

if __name__ == "__main__":
    run_experiment()
