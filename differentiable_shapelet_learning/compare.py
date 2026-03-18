import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import ShapeletNetwork, MLPBaseline, Conv1dBaseline
import os

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, lr, epochs=50, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dl_train:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y_test).float().mean().item()
    return accuracy

def objective(trial, model_type):
    X_train, y_train, X_test, y_test = get_data()
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    if model_type == "shapelet":
        model = ShapeletNetwork(in_channels=1, num_shapelets=40, shapelet_length=10, num_classes=10)
    elif model_type == "mlp":
        model = MLPBaseline(input_dim=40, hidden_dim=128, num_classes=10)
    elif model_type == "conv1d":
        model = Conv1dBaseline(in_channels=1, num_filters=32, kernel_size=5, num_classes=10)

    # Use fewer epochs for tuning
    train_model(model, X_train, y_train, lr, epochs=20)
    accuracy = evaluate_model(model, X_test, y_test)
    return accuracy

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()
    model_types = ["mlp", "conv1d", "shapelet"]
    results = {}

    for mt in model_types:
        if os.path.exists(f"differentiable_shapelet_learning/results_{mt}.txt"):
            with open(f"differentiable_shapelet_learning/results_{mt}.txt", "r") as f:
                line = f.read().strip()
                # format: accuracy={res['accuracy']}, best_lr={res['best_lr']}
                parts = line.split(", ")
                accuracy = float(parts[0].split("=")[1])
                best_lr = float(parts[1].split("=")[1])
                results[mt] = {"accuracy": accuracy, "best_lr": best_lr}
            print(f"Loaded results for {mt}: {results[mt]}")
            continue

        print(f"Tuning {mt}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, mt), n_trials=5)

        best_lr = study.best_params["lr"]
        print(f"Best LR for {mt}: {best_lr}")

        # Final train
        if mt == "shapelet":
            model = ShapeletNetwork(in_channels=1, num_shapelets=40, shapelet_length=10, num_classes=10)
        elif mt == "mlp":
            model = MLPBaseline(input_dim=40, hidden_dim=128, num_classes=10)
        elif mt == "conv1d":
            model = Conv1dBaseline(in_channels=1, num_filters=32, kernel_size=5, num_classes=10)

        train_model(model, X_train, y_train, best_lr, epochs=30)
        accuracy = evaluate_model(model, X_test, y_test)
        results[mt] = {"accuracy": accuracy, "best_lr": best_lr}
        print(f"Final accuracy for {mt}: {accuracy}")

        with open(f"differentiable_shapelet_learning/results_{mt}.txt", "w") as f:
            f.write(f"accuracy={accuracy}, best_lr={best_lr}\n")

        if mt == "shapelet":
            # Save learned shapelets
            shapelets = model.shapelet_layer.shapelets.detach().cpu().numpy()
            np.save("differentiable_shapelet_learning/learned_shapelets.npy", shapelets)

            # Plot some shapelets
            plt.figure(figsize=(10, 6))
            for i in range(min(16, shapelets.shape[0])):
                plt.subplot(4, 4, i+1)
                plt.plot(shapelets[i, 0, :])
                plt.axis('off')
            plt.suptitle("Learned Shapelets")
            plt.savefig("differentiable_shapelet_learning/learned_shapelets.png")
            plt.close()

    # Plot comparison
    plt.figure(figsize=(8, 6))
    plt.bar(results.keys(), [r["accuracy"] for r in results.values()])
    plt.ylabel("Test Accuracy")
    plt.title("Model Comparison on MNIST-1D")
    plt.savefig("differentiable_shapelet_learning/comparison.png")
    plt.close()

    with open("differentiable_shapelet_learning/results.txt", "w") as f:
        for mt, res in results.items():
            f.write(f"{mt}: accuracy={res['accuracy']}, best_lr={res['best_lr']}\n")

if __name__ == "__main__":
    run_experiment()
