import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import FrenetSerretAugmentedMLP, BaselineMLP
import matplotlib.pyplot as plt
import sys
import json

# Standard print with flush
def print_flush(msg):
    print(msg)
    sys.stdout.flush()

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

def train_model(model, train_loader, test_loader, lr, epochs=40):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        train_accuracies.append(train_acc)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100. * correct / total
        test_accuracies.append(test_acc)

    return train_accuracies, test_accuracies

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    if model_type == "augmented":
        model = FrenetSerretAugmentedMLP()
    else:
        model = BaselineMLP()

    _, test_accs = train_model(model, train_loader, test_loader, lr, epochs=15)
    score = max(test_accs)
    return score

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    # Tuning
    print_flush("Tuning Augmented MLP...")
    study_aug = optuna.create_study(direction="maximize")
    study_aug.optimize(lambda t: objective(t, "augmented", X_train, y_train, X_test, y_test), n_trials=10)
    best_lr_aug = study_aug.best_params["lr"]
    print_flush(f"Best LR for Augmented MLP: {best_lr_aug}")

    print_flush("Tuning Baseline MLP...")
    study_base = optuna.create_study(direction="maximize")
    study_base.optimize(lambda t: objective(t, "baseline", X_train, y_train, X_test, y_test), n_trials=10)
    best_lr_base = study_base.best_params["lr"]
    print_flush(f"Best LR for Baseline MLP: {best_lr_base}")

    # Final evaluation with multiple seeds
    seeds = [42, 43, 44, 45, 46]
    aug_final_accs = []
    base_final_accs = []

    aug_histories = []
    base_histories = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        print_flush(f"Evaluating Augmented MLP with seed {seed}...")
        model_aug = FrenetSerretAugmentedMLP()
        _, test_accs_aug = train_model(model_aug, train_loader, test_loader, best_lr_aug, epochs=50)
        aug_final_accs.append(test_accs_aug[-1])
        aug_histories.append(test_accs_aug)

        torch.manual_seed(seed)
        np.random.seed(seed)
        print_flush(f"Evaluating Baseline MLP with seed {seed}...")
        model_base = BaselineMLP()
        _, test_accs_base = train_model(model_base, train_loader, test_loader, best_lr_base, epochs=50)
        base_final_accs.append(test_accs_base[-1])
        base_histories.append(test_accs_base)

    results = {
        "best_lr_aug": best_lr_aug,
        "best_lr_base": best_lr_base,
        "aug_acc_mean": np.mean(aug_final_accs),
        "aug_acc_std": np.std(aug_final_accs),
        "base_acc_mean": np.mean(base_final_accs),
        "base_acc_std": np.std(base_final_accs),
        "aug_final_accs": aug_final_accs,
        "base_final_accs": base_final_accs
    }

    print_flush("\nResults Summary:")
    print_flush(f"Augmented MLP: {results['aug_acc_mean']:.2f}% +/- {results['aug_acc_std']:.2f}%")
    print_flush(f"Baseline MLP: {results['base_acc_mean']:.2f}% +/- {results['base_acc_std']:.2f}%")

    with open("differentiable_frenet_serret_experiment/results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Plot results
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, 51)

    aug_histories = np.array(aug_histories)
    base_histories = np.array(base_histories)

    plt.plot(epochs, aug_histories.mean(0), label="Frenet-Serret Augmented MLP")
    plt.fill_between(epochs, aug_histories.mean(0) - aug_histories.std(0), aug_histories.mean(0) + aug_histories.std(0), alpha=0.2)

    plt.plot(epochs, base_histories.mean(0), label="Baseline MLP")
    plt.fill_between(epochs, base_histories.mean(0) - base_histories.std(0), base_histories.mean(0) + base_histories.std(0), alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Frenet-Serret Augmented vs Baseline MLP on mnist1d")
    plt.legend()
    plt.grid(True)
    plt.savefig("differentiable_frenet_serret_experiment/comparison.png")
