import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import os
import sys
import matplotlib.pyplot as plt
from mnist1d.data import get_dataset_args, make_dataset

# Add current directory to path to import features
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import get_hoc_features

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_data():
    print("Loading mnist1d dataset...")
    args = get_dataset_args()
    data = make_dataset(args)

    # data is a dict with 'x', 'y', 'x_test', 'y_test'
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    return X_train, y_train, X_test, y_test

def shift_data(x, shift):
    return torch.roll(x, shifts=shift, dims=1)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear(x)

def train_model(model, X_train, y_train, X_val, y_val, lr, weight_decay, epochs=500, batch_size=256):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_preds = val_outputs.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc

    return best_val_acc

def objective_mlp(trial, X_train, y_train, X_val, y_val):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256)

    model = MLP(40, hidden_dim, 10)
    return train_model(model, X_train, y_train, X_val, y_val, lr, weight_decay, epochs=100)

def objective_linear(trial, X_train, y_train, X_val, y_val):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

    model = LinearModel(X_train.shape[1], 10)
    return train_model(model, X_train, y_train, X_val, y_val, lr, weight_decay, epochs=100)

def main():
    X_train_raw, y_train, X_test_raw, y_test = get_data()

    # Split train into train and val
    val_size = 500
    X_val_raw = X_train_raw[:val_size]
    y_val = y_train[:val_size]
    X_train_raw = X_train_raw[val_size:]
    y_train = y_train[val_size:]

    print(f"HOC Feature extraction...")
    X_train_hoc = get_hoc_features(X_train_raw)
    X_val_hoc = get_hoc_features(X_val_raw)
    X_test_hoc = get_hoc_features(X_test_raw)
    print(f"HOC Feature dim: {X_train_hoc.shape[1]}")

    # 1. Tune MLP
    print("\nTuning MLP...")
    study_mlp = optuna.create_study(direction="maximize")
    study_mlp.optimize(lambda t: objective_mlp(t, X_train_raw, y_train, X_val_raw, y_val), n_trials=10)
    best_mlp_params = study_mlp.best_params
    print(f"Best MLP params: {best_mlp_params}")

    # 2. Tune Linear
    print("\nTuning Linear Baseline...")
    study_linear = optuna.create_study(direction="maximize")
    study_linear.optimize(lambda t: objective_linear(t, X_train_raw, y_train, X_val_raw, y_val), n_trials=10)
    best_linear_params = study_linear.best_params
    print(f"Best Linear params: {best_linear_params}")

    # 3. Tune HOCNet
    print("\nTuning HOCNet...")
    study_hoc = optuna.create_study(direction="maximize")
    study_hoc.optimize(lambda t: objective_linear(t, X_train_hoc, y_train, X_val_hoc, y_val), n_trials=10)
    best_hoc_params = study_hoc.best_params
    print(f"Best HOCNet params: {best_hoc_params}")

    # Final Training and Evaluation
    print("\nFinal Training...")
    models = {
        "Linear": (LinearModel(40, 10), X_train_raw, X_test_raw, best_linear_params),
        "MLP": (MLP(40, best_mlp_params['hidden_dim'], 10), X_train_raw, X_test_raw, best_mlp_params),
        "HOCNet": (LinearModel(X_train_hoc.shape[1], 10), X_train_hoc, X_test_hoc, best_hoc_params)
    }

    results = {}
    for name, (model, xt, xtest, params) in models.items():
        print(f"  Training {name}...")
        train_model(model, xt, y_train, X_val_raw if name != "HOCNet" else X_val_hoc, y_val,
                    params['lr'], params['weight_decay'], epochs=200)

        model.eval()
        with torch.no_grad():
            # Original test accuracy
            outputs = model(xtest)
            preds = outputs.argmax(dim=1)
            acc = (preds == y_test).float().mean().item()

            # Shifted test accuracy
            # Apply random shifts
            acc_shifted_sum = 0
            num_shifts = 5
            for s in range(1, num_shifts + 1):
                shift_val = s * 5
                X_test_shifted_raw = shift_data(X_test_raw, shift_val)
                if name == "HOCNet":
                    xtest_shifted = get_hoc_features(X_test_shifted_raw)
                else:
                    xtest_shifted = X_test_shifted_raw

                outputs_s = model(xtest_shifted)
                preds_s = outputs_s.argmax(dim=1)
                acc_s = (preds_s == y_test).float().mean().item()
                acc_shifted_sum += acc_s

            acc_shifted = acc_shifted_sum / num_shifts
            results[name] = {"Original": acc, "Shifted": acc_shifted}
            print(f"    {name} Acc: {acc:.4f}, Shifted Acc: {acc_shifted:.4f}")

    # Save results
    with open("hoc_network_experiment/results.txt", "w") as f:
        f.write("Model,Original Accuracy,Shifted Accuracy\n")
        for name, res in results.items():
            f.write(f"{name},{res['Original']:.4f},{res['Shifted']:.4f}\n")

    # Plot results
    labels = list(results.keys())
    orig_accs = [results[l]['Original'] for l in labels]
    shift_accs = [results[l]['Shifted'] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, orig_accs, width, label='Original')
    ax.bar(x + width/2, shift_accs, width, label='Shifted')

    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Model and Test Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.savefig("hoc_network_experiment/comparison.png")
    plt.close()

if __name__ == "__main__":
    main()
