import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import optuna
from model import BaselineMLP, LMomentAugmentedMLP
import numpy as np
import time

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.int64)
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.int64)
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=50, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    return accuracy

def objective_baseline(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    X_train, y_train, X_test, y_test = get_data()
    model = BaselineMLP()
    acc = train_model(model, X_train, y_train, X_test, y_test, lr, epochs=30)
    return acc

def objective_lmoment(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    X_train, y_train, X_test, y_test = get_data()
    model = LMomentAugmentedMLP(window_size=10, stride=5) # More windows
    acc = train_model(model, X_train, y_train, X_test, y_test, lr, epochs=30)
    return acc

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    print("Tuning Baseline MLP...")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(objective_baseline, n_trials=12)
    best_lr_baseline = study_baseline.best_params["lr"]

    print("Tuning L-Moment Augmented MLP...")
    study_lmoment = optuna.create_study(direction="maximize")
    study_lmoment.optimize(objective_lmoment, n_trials=12)
    best_lr_lmoment = study_lmoment.best_params["lr"]

    print(f"Best LR Baseline: {best_lr_baseline}")
    print(f"Best LR L-Moment: {best_lr_lmoment}")

    # Final evaluation
    baseline_accs = []
    lmoment_accs = []
    for i in range(5):
        print(f"Final Run {i+1}/5...")
        model_b = BaselineMLP()
        acc_b = train_model(model_b, X_train, y_train, X_test, y_test, best_lr_baseline, epochs=100)
        baseline_accs.append(acc_b)

        model_l = LMomentAugmentedMLP(window_size=10, stride=5)
        acc_l = train_model(model_l, X_train, y_train, X_test, y_test, best_lr_lmoment, epochs=100)
        lmoment_accs.append(acc_l)

    print(f"Baseline Accuracy: {np.mean(baseline_accs):.4f} +/- {np.std(baseline_accs):.4f}")
    print(f"L-Moment Accuracy: {np.mean(lmoment_accs):.4f} +/- {np.std(lmoment_accs):.4f}")

    with open("differentiable_l_moments_experiment/results.txt", "w") as f:
        f.write(f"Best LR Baseline: {best_lr_baseline}\n")
        f.write(f"Best LR L-Moment: {best_lr_lmoment}\n")
        f.write(f"Baseline Accuracies: {baseline_accs}\n")
        f.write(f"Baseline Mean Accuracy: {np.mean(baseline_accs):.4f} +/- {np.std(baseline_accs):.4f}\n")
        f.write(f"L-Moment Accuracies: {lmoment_accs}\n")
        f.write(f"L-Moment Mean Accuracy: {np.mean(lmoment_accs):.4f} +/- {np.std(lmoment_accs):.4f}\n")
