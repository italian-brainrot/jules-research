import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import os
import sys

# Add current directory to path to allow imports from continuous_embedding_experiment
sys.path.append(os.getcwd())
from continuous_embedding_experiment.layers import FourierFeatures, LinearInterpolationEmbedding, DACE

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data():
    if hasattr(get_data, "cache"):
        return get_data.cache
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X = torch.tensor(data['x'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    # Split X, y into train and val
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    get_data.cache = (X_train, y_train, X_val, y_val, X_test, y_test)
    return get_data.cache

class MLP(nn.Module):
    def __init__(self, input_layer, input_dim=40, hidden_dim=256, output_dim=10):
        super(MLP, self).__init__()
        self.input_layer = input_layer

        # Determine the dimension after the input layer
        with torch.no_grad():
            dummy = torch.zeros(1, input_dim)
            out = self.input_layer(dummy)
            post_input_dim = out.shape[1]

        self.net = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.input_layer(x)
        return self.net(x)

def train_eval(lr, epochs, method="raw", method_params=None, is_final=False):
    if method_params is None:
        method_params = {}
    X_train, y_train, X_val, y_val, X_test, y_test = get_data()
    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)

    input_dim = 40
    num_embeddings = 32
    embedding_dim = 16

    if method == "raw":
        input_layer = nn.Identity()
    elif method == "fourier":
        input_layer = FourierFeatures(input_dim, input_dim * embedding_dim, sigma=method_params.get("sigma", 1.0))
    elif method == "line":
        input_layer = LinearInterpolationEmbedding(input_dim, num_embeddings, embedding_dim)
    elif method == "dace":
        input_layer = DACE(input_dim, num_embeddings, embedding_dim, gamma=method_params.get("gamma", 10.0))
    elif method == "mcce":
        input_layer = DACE(input_dim, num_embeddings, embedding_dim, gamma=method_params.get("gamma", 10.0))
    else:
        raise ValueError(f"Unknown method {method}")

    model = MLP(input_layer).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for inputs, targets in dl_train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)

            if method == "mcce":
                smoothness_weight = method_params.get("smoothness_weight", 0.01)
                loss += smoothness_weight * input_layer.get_smoothness_loss()

            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        if is_final:
            eval_X, eval_y = X_test, y_test
        else:
            eval_X, eval_y = X_val, y_val

        test_logits = model(eval_X.to(device))
        preds = test_logits.argmax(dim=1)
        acc = (preds == eval_y.to(device)).float().mean().item()

    return acc

def objective(trial, method):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    params = {}
    if method == "fourier":
        params["sigma"] = trial.suggest_float("sigma", 0.01, 10.0, log=True)
    elif method == "dace" or method == "mcce":
        params["gamma"] = trial.suggest_float("gamma", 0.1, 100.0, log=True)
        if method == "mcce":
            params["smoothness_weight"] = trial.suggest_float("smoothness_weight", 1e-4, 1.0, log=True)

    # Tuning epochs
    epochs = 30
    return train_eval(lr, epochs, method, params, is_final=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    n_trials = 10
    epochs = 50
    if args.smoke_test:
        n_trials = 1
        epochs = 1

    methods = ["raw", "fourier", "line", "dace", "mcce"]
    results = {}

    for method in methods:
        print(f"Tuning {method}...")
        study = optuna.create_study(direction="maximize")
        if args.smoke_test:
             study.optimize(lambda t: train_eval(1e-3, 1, method, is_final=False), n_trials=n_trials)
        else:
             study.optimize(lambda t: objective(t, method), n_trials=n_trials)

        best_params = study.best_params
        print(f"Training final {method} with best params: {best_params}")
        final_acc = train_eval(best_params.get("lr", 1e-3), epochs, method, best_params, is_final=True)
        results[method] = {"best_params": best_params, "final_acc": final_acc}

    # Save results
    os.makedirs("continuous_embedding_experiment", exist_ok=True)
    with open("continuous_embedding_experiment/results.txt", "w") as f:
        for method, res in results.items():
            f.write(f"Method: {method}\n")
            f.write(f"Best Params: {res['best_params']}\n")
            f.write(f"Final Test Accuracy: {res['final_acc']:.4f}\n")
            f.write("-" * 20 + "\n")

    # Generate a simple bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), [res['final_acc'] for res in results.values()])
    plt.ylabel("Test Accuracy")
    plt.title("Comparison of Embedding Methods on MNIST-1D")
    plt.savefig("continuous_embedding_experiment/accuracy_comparison.png")

    print("\nFinal Results:")
    for method, res in results.items():
        print(f"{method}: {res['final_acc']:.4f}")
