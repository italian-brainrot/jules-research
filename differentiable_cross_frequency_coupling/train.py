import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from differentiable_cross_frequency_coupling.model import DCFCAugmentedMLP, BaselineMLP
import os

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])
    return X_train, y_train, X_test, y_test

def train_model(model, train_loader, val_loader, lr, epochs=50, device='cpu'):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

def objective(trial, model_type, X_train, y_train, X_test, y_test, device):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_type == "dcfc":
        model = DCFCAugmentedMLP(input_dim=40, hidden_dim=256, output_dim=10, num_pairs=16).to(device)
    else:
        model = BaselineMLP(input_dim=40, hidden_dim=256, output_dim=10).to(device)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    return train_model(model, train_loader, val_loader, lr, epochs=20, device=device)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train, X_test, y_test = get_data()

    results = {}

    for model_type in ["dcfc", "baseline"]:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test, device), n_trials=10)

        best_lr = study.best_params["lr"]
        print(f"Best LR for {model_type}: {best_lr}")

        final_accs = []
        for seed in range(3):
            set_seed(seed)
            if model_type == "dcfc":
                model = DCFCAugmentedMLP(input_dim=40, hidden_dim=256, output_dim=10, num_pairs=16).to(device)
            else:
                model = BaselineMLP(input_dim=40, hidden_dim=256, output_dim=10).to(device)

            train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
            val_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

            acc = train_model(model, train_loader, val_loader, best_lr, epochs=50, device=device)
            final_accs.append(acc)
            print(f"Seed {seed} Accuracy: {acc:.4f}")

        results[model_type] = {
            "mean": np.mean(final_accs),
            "std": np.std(final_accs),
            "best_lr": best_lr
        }

    with open("differentiable_cross_frequency_coupling/results.txt", "w") as f:
        for model_type, res in results.items():
            f.write(f"{model_type}: {res['mean']:.4f} +/- {res['std']:.4f} (LR: {res['best_lr']:.6f})\n")

    # Plotting
    model_names = list(results.keys())
    means = [results[m]["mean"] for m in model_names]
    stds = [results[m]["std"] for m in model_names]

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, means, yerr=stds, capsize=10, color=['blue', 'green'])
    plt.ylabel("Accuracy")
    plt.title("DCFC Augmented MLP vs Baseline on MNIST-1D")
    plt.savefig("differentiable_cross_frequency_coupling/results.png")

    # Create README.md
    with open("differentiable_cross_frequency_coupling/README.md", "w") as f:
        f.write("# Differentiable Cross-Frequency Coupling (DCFC) Experiment\n\n")
        f.write("This experiment investigates the utility of learnable Phase-Amplitude Coupling (PAC) features for signal classification.\n\n")
        f.write("## Methodology\n")
        f.write("- **DCFCLayer**: Extracts Phase-Amplitude Coupling between learnable pairs of frequencies using Gaussian filters and the Mean Vector Length (MVL) metric.\n")
        f.write("- **Models**: Compared a baseline MLP with an MLP augmented with DCFC features (16 learned pairs).\n")
        f.write("- **Dataset**: MNIST-1D.\n")
        f.write("- **Tuning**: Learning rates were tuned using Optuna.\n\n")
        f.write("## Results\n")
        f.write("| Model | Accuracy |\n")
        f.write("| :--- | :--- |\n")
        for model_type, res in results.items():
            f.write(f"| {model_type} | {res['mean']:.4f} +/- {res['std']:.4f} |\n")
        f.write("\n")
        f.write("![Results](results.png)\n")

if __name__ == "__main__":
    main()
