import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from arnoldi_feature_layer_experiment.model import ArnoldiMLP, BaselineMLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test'])
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, lr, epochs=30, batch_size=128):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    test_accs = []
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
            acc = (outputs.argmax(1) == y_test).float().mean().item()
            test_accs.append(acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Test Acc: {acc:.4f}")

    return test_accs

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = 128

    if model_type == "baseline":
        model = BaselineMLP(40, hidden_dim, 10, num_layers=3)
    else:
        k = 3
        num_heads = 2
        model = ArnoldiMLP(40, k, num_heads, hidden_dim, 10, num_layers=3)

    test_accs = train_model(model, X_train, y_train, X_test, y_test, lr, epochs=15)
    return max(test_accs)

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    print("--- Tuning Baseline MLP ---")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(lambda trial: objective(trial, "baseline", X_train, y_train, X_test, y_test), n_trials=5)
    best_lr_baseline = study_baseline.best_params["lr"]

    print("--- Tuning Arnoldi MLP ---")
    study_arnoldi = optuna.create_study(direction="maximize")
    study_arnoldi.optimize(lambda trial: objective(trial, "arnoldi", X_train, y_train, X_test, y_test), n_trials=5)
    best_lr_arnoldi = study_arnoldi.best_params["lr"]

    print(f"Best LR Baseline: {best_lr_baseline:.6f}")
    print(f"Best LR Arnoldi: {best_lr_arnoldi:.6f}")

    seeds = [42, 43]
    baseline_final = []
    arnoldi_final = []

    print("\n--- Final Evaluation ---")
    for seed in seeds:
        print(f"Seed {seed}")
        torch.manual_seed(seed)
        model_baseline = BaselineMLP(40, 128, 10, num_layers=3)
        accs = train_model(model_baseline, X_train, y_train, X_test, y_test, best_lr_baseline, epochs=40)
        baseline_final.append(accs)

        torch.manual_seed(seed)
        model_arnoldi = ArnoldiMLP(40, 3, 2, 128, 10, num_layers=3)
        accs = train_model(model_arnoldi, X_train, y_train, X_test, y_test, best_lr_arnoldi, epochs=40)
        arnoldi_final.append(accs)

    baseline_mean = np.mean(baseline_final, axis=0)
    arnoldi_mean = np.mean(arnoldi_final, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(baseline_mean, label="Baseline MLP")
    plt.plot(arnoldi_mean, label="Arnoldi-MLP (Augmented)")
    plt.title("Arnoldi Feature Layer vs Baseline MLP (MNIST-1D)")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("arnoldi_feature_layer_experiment/comparison.png")

    results_text = (
        f"Best LR Baseline: {best_lr_baseline:.6f}\n"
        f"Best LR Arnoldi: {best_lr_arnoldi:.6f}\n"
        f"Final Accuracy Baseline: {baseline_mean[-1]:.4f} +/- {np.std([a[-1] for a in baseline_final]):.4f}\n"
        f"Final Accuracy Arnoldi: {arnoldi_mean[-1]:.4f} +/- {np.std([a[-1] for a in arnoldi_final]):.4f}\n"
    )
    print("\n" + results_text)
    with open("arnoldi_feature_layer_experiment/results.txt", "w") as f:
        f.write(results_text)

if __name__ == "__main__":
    run_experiment()
