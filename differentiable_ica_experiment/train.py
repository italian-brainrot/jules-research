import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import optuna
import numpy as np
import os
import matplotlib.pyplot as plt
from differentiable_ica_experiment.model import ICAClassifier

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test'])

    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, lr, epochs=50, batch_size=256):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(model, X_test, y_test, batch_size=256):
    model.eval()
    dl_test = TensorDataLoader((X_test, y_test), batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dl_test:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = 10

    model = ICAClassifier(input_dim, hidden_dim, output_dim, mode=model_type, ica_iters=10)

    train_model(model, X_train, y_train, lr, epochs=20)
    accuracy = evaluate_model(model, X_test, y_test)
    return accuracy

def main():
    X_train, y_train, X_test, y_test = get_data()

    results = {}
    best_lrs = {}

    model_types = ["baseline", "pca", "ica", "pca_aug", "ica_aug"]

    for model_type in model_types:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=5)

        best_lr = study.best_params["lr"]
        best_lrs[model_type] = best_lr
        print(f"Best LR for {model_type}: {best_lr}")

        # Final evaluation with 5 seeds
        accs = []
        for seed in range(5):
            torch.manual_seed(seed)
            input_dim = X_train.shape[1]
            hidden_dim = 128
            output_dim = 10

            model = ICAClassifier(input_dim, hidden_dim, output_dim, mode=model_type, ica_iters=10)

            train_model(model, X_train, y_train, best_lr, epochs=50)
            acc = evaluate_model(model, X_test, y_test)
            accs.append(acc)
            print(f"{model_type} Seed {seed} accuracy: {acc:.2f}%")

        results[model_type] = (np.mean(accs), np.std(accs))
        print(f"{model_type} final accuracy: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")

    with open("differentiable_ica_experiment/results.txt", "w") as f:
        f.write("Differentiable ICA Experiment Results (Enhanced)\n")
        f.write("================================================\n")
        for model_type, (mean, std) in results.items():
            f.write(f"{model_type}: {mean:.2f}% +/- {std:.2f}% (Best LR: {best_lrs[model_type]:.6f})\n")

    # Plot learned components for ICA and PCA
    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = 10

    pca_model = ICAClassifier(input_dim, hidden_dim, output_dim, mode='pca')
    train_model(pca_model, X_train, y_train, best_lrs["pca"], epochs=5)

    ica_model = ICAClassifier(input_dim, hidden_dim, output_dim, mode='ica', ica_iters=20)
    train_model(ica_model, X_train, y_train, best_lrs["ica"], epochs=5)

    plt.figure(figsize=(15, 8))
    sample_x = X_train[:5]
    pca_out = pca_model.pre(sample_x).detach().numpy()
    ica_out = ica_model.pre(sample_x).detach().numpy()

    for i in range(5):
        plt.subplot(3, 5, i+1)
        plt.plot(sample_x[i].numpy())
        if i == 0: plt.ylabel("Original")
        plt.subplot(3, 5, i+6)
        plt.plot(pca_out[i])
        if i == 0: plt.ylabel("PCA Features")
        plt.subplot(3, 5, i+11)
        plt.plot(ica_out[i])
        if i == 0: plt.ylabel("ICA Features")

    plt.tight_layout()
    plt.savefig("differentiable_ica_experiment/features_enhanced.png")

if __name__ == "__main__":
    main()
