import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from differentiable_gabor_filter_experiment.model import BaselineConvMLP, GaborConvMLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test'])
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=50, batch_size=128):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    train_losses = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(dl_train))

        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            acc = (outputs.argmax(1) == y_test).float().mean().item()
            test_accs.append(acc)

    return train_losses, test_accs

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)

    input_dim = 40
    hidden_dim = 256
    output_dim = 10
    kernel_size = 15
    num_filters = 16

    if model_type == "baseline":
        model = BaselineConvMLP(input_dim, hidden_dim, output_dim, kernel_size, num_filters)
    else:
        model = GaborConvMLP(input_dim, hidden_dim, output_dim, kernel_size, num_filters)

    _, test_accs = train_model(model, X_train, y_train, X_test, y_test, lr, epochs=20)
    return max(test_accs)

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    # Tune baseline
    print("Tuning baseline...")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(lambda trial: objective(trial, "baseline", X_train, y_train, X_test, y_test), n_trials=15)
    best_lr_baseline = study_baseline.best_params["lr"]

    # Tune Gabor
    print("Tuning Gabor...")
    study_gabor = optuna.create_study(direction="maximize")
    study_gabor.optimize(lambda trial: objective(trial, "gabor", X_train, y_train, X_test, y_test), n_trials=15)
    best_lr_gabor = study_gabor.best_params["lr"]

    print(f"Best LR Baseline: {best_lr_baseline}")
    print(f"Best LR Gabor: {best_lr_gabor}")

    # Final evaluation
    print("Starting final evaluation...")
    seeds = [42, 43, 44]
    baseline_results = []
    gabor_results = []

    input_dim = 40
    hidden_dim = 256
    output_dim = 10
    kernel_size = 15
    num_filters = 16

    for seed in seeds:
        print(f"Seed {seed}...")
        torch.manual_seed(seed)
        model_baseline = BaselineConvMLP(input_dim, hidden_dim, output_dim, kernel_size, num_filters)
        _, test_accs_baseline = train_model(model_baseline, X_train, y_train, X_test, y_test, best_lr_baseline, epochs=50)
        baseline_results.append(test_accs_baseline)

        torch.manual_seed(seed)
        model_gabor = GaborConvMLP(input_dim, hidden_dim, output_dim, kernel_size, num_filters)
        _, test_accs_gabor = train_model(model_gabor, X_train, y_train, X_test, y_test, best_lr_gabor, epochs=50)
        gabor_results.append(test_accs_gabor)

    baseline_mean = np.mean(baseline_results, axis=0)
    gabor_mean = np.mean(gabor_results, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(baseline_mean, label=f"Baseline ConvMLP (LR={best_lr_baseline:.4f})")
    plt.plot(gabor_mean, label=f"Gabor ConvMLP (LR={best_lr_gabor:.4f})")
    plt.title("Test Accuracy Comparison on MNIST-1D")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("differentiable_gabor_filter_experiment/comparison.png")

    print(f"Final Test Accuracy (Baseline): {baseline_mean[-1]:.4f} +/- {np.std([r[-1] for r in baseline_results]):.4f}")
    print(f"Final Test Accuracy (Gabor): {gabor_mean[-1]:.4f} +/- {np.std([r[-1] for r in gabor_results]):.4f}")

    with open("differentiable_gabor_filter_experiment/results.txt", "w") as f:
        f.write(f"Best LR Baseline: {best_lr_baseline}\n")
        f.write(f"Best LR Gabor: {best_lr_gabor}\n")
        f.write(f"Final Test Accuracy (Baseline): {baseline_mean[-1]:.4f} +/- {np.std([r[-1] for r in baseline_results]):.4f}\n")
        f.write(f"Final Test Accuracy (Gabor): {gabor_mean[-1]:.4f} +/- {np.std([r[-1] for r in gabor_results]):.4f}\n")

if __name__ == "__main__":
    run_experiment()
