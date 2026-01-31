import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
import os
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import AdaptiveMLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, X_test, y_test, use_mixture=True, lr=1e-3, epochs=20, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptiveMLP(40, [256, 256], 10, use_mixture=use_mixture).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = TensorDataLoader((X_train.to(device), y_train.to(device)), batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "test_acc": [], "mixture_weights": [], "omegas": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            outputs = model(X_test.to(device))
            acc = (outputs.argmax(1) == y_test.to(device)).float().mean().item()

        history["train_loss"].append(avg_loss)
        history["test_acc"].append(acc)
        if use_mixture:
            history["mixture_weights"].append(model.get_mixture_weights())
            history["omegas"].append(model.get_omegas())

    return model, history

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    print("Tuning Baseline (ReLU)...")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(lambda t: max(train_model(X_train, y_train, X_test, y_test, use_mixture=False, lr=t.suggest_float("lr", 1e-4, 1e-2, log=True), epochs=10)[1]["test_acc"]), n_trials=15)

    print("Tuning Adaptive Mixture (AMA)...")
    study_ama = optuna.create_study(direction="maximize")
    study_ama.optimize(lambda t: max(train_model(X_train, y_train, X_test, y_test, use_mixture=True, lr=t.suggest_float("lr", 1e-4, 1e-2, log=True), epochs=10)[1]["test_acc"]), n_trials=15)

    best_lr_baseline = study_baseline.best_params["lr"]
    best_lr_ama = study_ama.best_params["lr"]

    print(f"Best LR Baseline: {best_lr_baseline}")
    print(f"Best LR AMA: {best_lr_ama}")

    print("Final training runs...")
    seeds = [42, 43, 44]
    baseline_accs = []
    ama_accs = []
    baseline_histories = []
    ama_histories = []

    for seed in seeds:
        torch.manual_seed(seed)
        _, h_b = train_model(X_train, y_train, X_test, y_test, use_mixture=False, lr=best_lr_baseline, epochs=30)
        baseline_histories.append(h_b)
        baseline_accs.append(max(h_b["test_acc"]))

        torch.manual_seed(seed)
        m_a, h_a = train_model(X_train, y_train, X_test, y_test, use_mixture=True, lr=best_lr_ama, epochs=30)
        ama_histories.append(h_a)
        ama_accs.append(max(h_a["test_acc"]))

    print(f"Baseline Mean Accuracy: {np.mean(baseline_accs):.4f}")
    print(f"AMA Mean Accuracy: {np.mean(ama_accs):.4f}")

    # Plot results
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    for h in baseline_histories:
        plt.plot(h["test_acc"], color='blue', alpha=0.3)
    plt.plot(np.mean([h["test_acc"] for h in baseline_histories], axis=0), label="Baseline (ReLU)", color='blue', linewidth=2)

    for h in ama_histories:
        plt.plot(h["test_acc"], color='orange', alpha=0.3)
    plt.plot(np.mean([h["test_acc"] for h in ama_histories], axis=0), label="Adaptive Mixture (AMA)", color='orange', linewidth=2)

    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    for h in baseline_histories:
        plt.plot(h["train_loss"], color='blue', alpha=0.3)
    plt.plot(np.mean([h["train_loss"] for h in baseline_histories], axis=0), label="Baseline (ReLU)", color='blue', linewidth=2)

    for h in ama_histories:
        plt.plot(h["train_loss"], color='orange', alpha=0.3)
    plt.plot(np.mean([h["train_loss"] for h in ama_histories], axis=0), label="Adaptive Mixture (AMA)", color='orange', linewidth=2)

    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("adaptive_activation_mixture_experiment/comparison.png")

    # Mixture weights evolution
    final_ama_history = ama_histories[0] # Take one seed
    weights_evo = np.array(final_ama_history["mixture_weights"]) # (epochs, num_layers, 4)
    num_layers = weights_evo.shape[1]

    plt.figure(figsize=(15, 5))
    labels = ['ReLU', 'Tanh', 'Sin', 'Identity']
    for i in range(num_layers):
        plt.subplot(1, num_layers, i+1)
        for j in range(4):
            plt.plot(weights_evo[:, i, j], label=labels[j])
        plt.title(f"Layer {i+1} Mixture Weights")
        plt.xlabel("Epoch")
        plt.ylabel("Weight")
        plt.legend()

    plt.tight_layout()
    plt.savefig("adaptive_activation_mixture_experiment/mixture_weights_evolution.png")

    # Omegas
    omegas_evo = np.array(final_ama_history["omegas"]) # (epochs, num_layers)
    plt.figure()
    for i in range(num_layers):
        plt.plot(omegas_evo[:, i], label=f"Layer {i+1}")
    plt.title("Sin Frequency (Omega) Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Omega")
    plt.legend()
    plt.savefig("adaptive_activation_mixture_experiment/omegas_evolution.png")

    with open("adaptive_activation_mixture_experiment/results.txt", "w") as f:
        f.write(f"Baseline Accuracy: {np.mean(baseline_accs):.4f} +/- {np.std(baseline_accs):.4f}\n")
        f.write(f"AMA Accuracy: {np.mean(ama_accs):.4f} +/- {np.std(ama_accs):.4f}\n")
        f.write(f"Best LR Baseline: {best_lr_baseline}\n")
        f.write(f"Best LR AMA: {best_lr_ama}\n")
        f.write(f"Final Weights Layer 1: {weights_evo[-1, 0, :]}\n")
        f.write(f"Final Weights Layer 2: {weights_evo[-1, 1, :]}\n")
        f.write(f"Final Omegas: {omegas_evo[-1, :]}\n")

if __name__ == "__main__":
    run_experiment()
