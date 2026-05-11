import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from differentiable_fisher_vector_experiment.model import BaselineMLP, DFVNet, DFVAugmentedMLP

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
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    input_dim = 40
    hidden_dim = 256
    output_dim = 10
    num_clusters = 8
    patch_size = 10
    stride = 5

    if model_type == "baseline":
        model = BaselineMLP(input_dim, hidden_dim, output_dim)
    elif model_type == "dfv":
        model = DFVNet(input_dim, num_clusters, patch_size, stride, hidden_dim, output_dim)
    else:
        model = DFVAugmentedMLP(input_dim, num_clusters, patch_size, stride, hidden_dim, output_dim)

    _, test_accs = train_model(model, X_train, y_train, X_test, y_test, lr, epochs=20)
    return max(test_accs)

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    models_to_test = ["baseline", "dfv", "dfv_augmented"]
    best_lrs = {}

    for model_type in models_to_test:
        print(f"Tuning {model_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_type, X_train, y_train, X_test, y_test), n_trials=10)
        best_lrs[model_type] = study.best_params["lr"]
        print(f"Best LR for {model_type}: {best_lrs[model_type]}")

    # Final evaluation
    print("Starting final evaluation...")
    seeds = [42, 43, 44]
    results = {m: [] for m in models_to_test}

    input_dim = 40
    hidden_dim = 256
    output_dim = 10
    num_clusters = 8
    patch_size = 10
    stride = 5

    for seed in seeds:
        print(f"Seed {seed}...")
        for model_type in models_to_test:
            torch.manual_seed(seed)
            if model_type == "baseline":
                model = BaselineMLP(input_dim, hidden_dim, output_dim)
            elif model_type == "dfv":
                model = DFVNet(input_dim, num_clusters, patch_size, stride, hidden_dim, output_dim)
            else:
                model = DFVAugmentedMLP(input_dim, num_clusters, patch_size, stride, hidden_dim, output_dim)

            _, test_accs = train_model(model, X_train, y_train, X_test, y_test, best_lrs[model_type], epochs=50)
            results[model_type].append(test_accs)

    plt.figure(figsize=(10, 6))
    for model_type in models_to_test:
        mean_accs = np.mean(results[model_type], axis=0)
        plt.plot(mean_accs, label=f"{model_type} (LR={best_lrs[model_type]:.4f})")

    plt.title("Test Accuracy Comparison on MNIST-1D")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("differentiable_fisher_vector_experiment/comparison.png")

    with open("differentiable_fisher_vector_experiment/results.txt", "w") as f:
        for model_type in models_to_test:
            final_accs = [r[-1] for r in results[model_type]]
            mean_acc = np.mean(final_accs)
            std_acc = np.std(final_accs)
            line = f"Final Test Accuracy ({model_type}): {mean_acc:.4f} +/- {std_acc:.4f}\n"
            print(line.strip())
            f.write(line)
            f.write(f"Best LR ({model_type}): {best_lrs[model_type]}\n")

if __name__ == "__main__":
    run_experiment()
