import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import numpy as np
import os
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import SFAMLP, BaselineMLP

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_data():
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr=1e-3, epochs=30, sfa_lambda=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_dl = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

    history = {'train_loss': [], 'test_acc': [], 'sfa_penalty': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_sfa = 0
        for inputs, targets in train_dl:
            optimizer.zero_grad()
            logits, sfa_penalty = model(inputs)
            loss = criterion(logits, targets)

            # Combine losses
            total_combined_loss = loss + sfa_lambda * sfa_penalty

            total_combined_loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_sfa += sfa_penalty.item()

        model.eval()
        with torch.no_grad():
            test_logits, _ = model(X_test)
            test_preds = torch.argmax(test_logits, dim=1)
            test_acc = (test_preds == y_test).float().mean().item()

        history['train_loss'].append(total_loss / len(train_dl))
        history['test_acc'].append(test_acc)
        history['sfa_penalty'].append(total_sfa / len(train_dl))

    return history, history['test_acc'][-1]

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_type == "sfa":
        sfa_lambda = trial.suggest_float("sfa_lambda", 1e-3, 10.0, log=True)
        model = SFAMLP(input_dim=40, hidden_dim=128, output_dim=10, num_layers=3)
        history, final_acc = train_model(model, X_train, y_train, X_test, y_test, lr=lr, epochs=20, sfa_lambda=sfa_lambda)
    else:
        model = BaselineMLP(input_dim=40, hidden_dim=128, output_dim=10, num_layers=3)
        history, final_acc = train_model(model, X_train, y_train, X_test, y_test, lr=lr, epochs=20, sfa_lambda=0.0)

    return final_acc

def main():
    X_train, y_train, X_test, y_test = get_data()

    print("Tuning Baseline MLP...")
    baseline_study = optuna.create_study(direction="maximize")
    baseline_study.optimize(lambda trial: objective(trial, "baseline", X_train, y_train, X_test, y_test), n_trials=10)
    best_baseline_lr = baseline_study.best_params['lr']

    print("Tuning SFA MLP...")
    sfa_study = optuna.create_study(direction="maximize")
    sfa_study.optimize(lambda trial: objective(trial, "sfa", X_train, y_train, X_test, y_test), n_trials=10)
    best_sfa_lr = sfa_study.best_params['lr']
    best_sfa_lambda = sfa_study.best_params['sfa_lambda']

    print(f"Best Baseline LR: {best_baseline_lr}")
    print(f"Best SFA LR: {best_sfa_lr}, SFA Lambda: {best_sfa_lambda}")

    # Train with best params and more epochs/seeds
    num_seeds = 3
    epochs = 40

    baseline_results = []
    sfa_results = []

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        print(f"Evaluating Baseline Seed {seed}...")
        model = BaselineMLP(num_layers=3)
        h, acc = train_model(model, X_train, y_train, X_test, y_test, lr=best_baseline_lr, epochs=epochs, sfa_lambda=0.0)
        baseline_results.append(h)

        print(f"Evaluating SFA Seed {seed}...")
        model = SFAMLP(num_layers=3)
        h, acc = train_model(model, X_train, y_train, X_test, y_test, lr=best_sfa_lr, epochs=epochs, sfa_lambda=best_sfa_lambda)
        sfa_results.append(h)

    # Calculate means
    def get_mean_acc(results):
        accs = np.array([r['test_acc'] for r in results])
        return accs.mean(axis=0), accs.std(axis=0)

    baseline_mean, baseline_std = get_mean_acc(baseline_results)
    sfa_mean, sfa_std = get_mean_acc(sfa_results)

    # Plot results
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, baseline_mean, label=f"Baseline (Final: {baseline_mean[-1]:.4f})")
    plt.fill_between(epochs_range, baseline_mean - baseline_std, baseline_mean + baseline_std, alpha=0.2)
    plt.plot(epochs_range, sfa_mean, label=f"SFA MLP (Final: {sfa_mean[-1]:.4f})")
    plt.fill_between(epochs_range, sfa_mean - sfa_std, sfa_mean + sfa_std, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("SFA MLP vs Baseline MLP on MNIST-1D")
    plt.legend()
    plt.grid(True)
    plt.savefig("differentiable_slow_feature_analysis_experiment/accuracy_plot.png")

    # Save results to file
    with open("differentiable_slow_feature_analysis_experiment/results.txt", "w") as f:
        f.write(f"Baseline Mean Accuracy: {baseline_mean[-1]:.4f} +/- {baseline_std[-1]:.4f}\n")
        f.write(f"SFA MLP Mean Accuracy: {sfa_mean[-1]:.4f} +/- {sfa_std[-1]:.4f}\n")
        f.write(f"Best Baseline LR: {best_baseline_lr}\n")
        f.write(f"Best SFA LR: {best_sfa_lr}, SFA Lambda: {best_sfa_lambda}\n")

if __name__ == "__main__":
    main()
