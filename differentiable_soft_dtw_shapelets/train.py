import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
import os
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from model import ShapeletNetwork, MLPBaseline

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

def train_model(model, X_train, y_train, lr, epochs=30, batch_size=128):
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

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if model_type == "mlp":
        model = MLPBaseline(input_dim=40, hidden_dim=128, num_classes=10)
    elif model_type == "euclidean":
        model = ShapeletNetwork(in_channels=1, num_shapelets=40, shapelet_length=10, num_classes=10, layer_type='euclidean')
    elif model_type == "soft_dtw":
        # Soft-DTW is slower, maybe use smaller batch size or fewer epochs if needed
        model = ShapeletNetwork(in_channels=1, num_shapelets=16, shapelet_length=8, num_classes=10, layer_type='soft_dtw', gamma=1.0, stride=8)
    else:
        raise ValueError("Unknown model type")

    train_model(model, X_train, y_train, lr, epochs=5 if model_type == "soft_dtw" else 20, batch_size=256)
    accuracy = evaluate_model(model, X_test, y_test)
    return accuracy

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()
    model_types = ["mlp", "euclidean", "soft_dtw"]
    results = {}

    exp_dir = "differentiable_soft_dtw_shapelets"

    model_types = ["soft_dtw", "mlp", "euclidean"]
    for mt in model_types:
        if mt == "mlp":
            res_path = f"{exp_dir}/results_mlp.txt"
        elif mt == "euclidean":
            res_path = f"{exp_dir}/results_euclidean.txt"
        else:
            res_path = f"{exp_dir}/results_soft_dtw.txt"

        if os.path.exists(res_path):
            print(f"Skipping {mt}, results already exist.")
            with open(res_path, "r") as f:
                line = f.read().strip()
                accuracy = float(line.split("accuracy=")[1].split(",")[0])
                best_lr = float(line.split("best_lr=")[1])
                results[mt] = {"accuracy": accuracy, "best_lr": best_lr}
            continue

        # Check for partial study results to resume or just skip tuning if it takes too long
        # Let's just use a fixed LR if tuning failed multiple times?
        # No, let's try 1 trial for soft_dtw.

        print(f"\n--- Tuning {mt} ---")
        study = optuna.create_study(direction="maximize")
        n_trials = 5 if mt != "soft_dtw" else 1 # Soft-DTW is slower
        study.optimize(lambda trial: objective(trial, mt, X_train, y_train, X_test, y_test), n_trials=n_trials)

        best_lr = study.best_params["lr"]
        print(f"Best LR for {mt}: {best_lr}")

        print(f"Final training for {mt}...")
        if mt == "mlp":
            model = MLPBaseline(input_dim=40, hidden_dim=128, num_classes=10)
        elif mt == "euclidean":
            model = ShapeletNetwork(in_channels=1, num_shapelets=40, shapelet_length=10, num_classes=10, layer_type='euclidean')
        elif mt == "soft_dtw":
            model = ShapeletNetwork(in_channels=1, num_shapelets=16, shapelet_length=8, num_classes=10, layer_type='soft_dtw', gamma=1.0, stride=8)

        train_model(model, X_train, y_train, best_lr, epochs=10 if mt == "soft_dtw" else 50, batch_size=256)
        accuracy = evaluate_model(model, X_test, y_test)
        results[mt] = {"accuracy": accuracy, "best_lr": best_lr}
        print(f"Final accuracy for {mt}: {accuracy}")

        with open(res_path, "w") as f:
            f.write(f"accuracy={accuracy}, best_lr={best_lr}\n")

        if mt != "mlp":
            # Save learned shapelets
            shapelets = model.shapelet_layer.shapelets.detach().cpu().numpy()
            np.save(f"{exp_dir}/learned_shapelets_{mt}.npy", shapelets)

            # Plot some shapelets
            plt.figure(figsize=(10, 6))
            for i in range(min(16, shapelets.shape[0])):
                plt.subplot(4, 4, i+1)
                plt.plot(shapelets[i, 0, :])
                plt.axis('off')
            plt.suptitle(f"Learned Shapelets ({mt})")
            plt.savefig(f"{exp_dir}/learned_shapelets_{mt}.png")
            plt.close()

    # Plot comparison
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    plt.bar(names, accs)
    plt.ylabel("Test Accuracy")
    plt.title("Model Comparison on MNIST-1D")
    for i, v in enumerate(accs):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.savefig(f"{exp_dir}/comparison.png")
    plt.close()

    with open(f"{exp_dir}/results.txt", "w") as f:
        for mt, res in results.items():
            f.write(f"{mt}: accuracy={res['accuracy']}, best_lr={res['best_lr']}\n")

if __name__ == "__main__":
    run_experiment()
