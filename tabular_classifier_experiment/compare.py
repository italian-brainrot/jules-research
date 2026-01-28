import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import numpy as np
import os
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from model import PrototypeClassifier, MLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y']).long()
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test']).long()
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, lr=1e-3, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    train_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            output = model(X_test)
            pred = output.argmax(dim=1)
            acc = (pred == y_test).float().mean().item()
            val_accs.append(acc)

    return train_losses, val_accs

def objective_mlp(trial, X_train, y_train, X_test, y_test):
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    n_layers = trial.suggest_int('n_layers', 1, 4)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    model = MLP(40, 10, hidden_dim, n_layers)
    _, val_accs = train_model(model, X_train, y_train, X_test, y_test, epochs=20, lr=lr)
    return max(val_accs)

def objective_pc(trial, X_train, y_train, X_test, y_test):
    n_prototypes = trial.suggest_int('n_prototypes', 5, 50)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    model = PrototypeClassifier(40, 10, n_prototypes)
    _, val_accs = train_model(model, X_train, y_train, X_test, y_test, epochs=20, lr=lr)
    return max(val_accs)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("Tuning MLP...")
    study_mlp = optuna.create_study(direction='maximize')
    study_mlp.optimize(lambda trial: objective_mlp(trial, X_train, y_train, X_test, y_test), n_trials=10)

    print("Tuning PrototypeClassifier...")
    study_pc = optuna.create_study(direction='maximize')
    study_pc.optimize(lambda trial: objective_pc(trial, X_train, y_train, X_test, y_test), n_trials=10)

    print(f"Best MLP Accuracy during tuning: {study_mlp.best_value}")
    print(f"Best PrototypeClassifier Accuracy during tuning: {study_pc.best_value}")

    # Train best models for longer
    print("Training best MLP...")
    best_mlp_params = study_mlp.best_params
    best_mlp = MLP(40, 10, best_mlp_params['hidden_dim'], best_mlp_params['n_layers'])
    mlp_train_loss, mlp_val_acc = train_model(best_mlp, X_train, y_train, X_test, y_test, epochs=50, lr=best_mlp_params['lr'])

    print("Training best PrototypeClassifier...")
    best_pc_params = study_pc.best_params
    best_pc = PrototypeClassifier(40, 10, best_pc_params['n_prototypes'])
    pc_train_loss, pc_val_acc = train_model(best_pc, X_train, y_train, X_test, y_test, epochs=50, lr=best_pc_params['lr'])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mlp_train_loss, label='MLP Train Loss')
    plt.plot(pc_train_loss, label='PrototypeClassifier Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mlp_val_acc, label='MLP Val Acc')
    plt.plot(pc_val_acc, label='PrototypeClassifier Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('tabular_classifier_experiment/results.png')

    with open('tabular_classifier_experiment/results.txt', 'w') as f:
        f.write(f"Best MLP Accuracy: {max(mlp_val_acc)}\n")
        f.write(f"Best MLP Params: {best_mlp_params}\n")
        f.write(f"Best PrototypeClassifier Accuracy: {max(pc_val_acc)}\n")
        f.write(f"Best PrototypeClassifier Params: {best_pc_params}\n")

    print("Results saved.")
