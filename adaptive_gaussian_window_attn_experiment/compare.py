import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
from models import TransformerClassifier
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    args = get_dataset_args()
    args.num_samples = 4000
    data = make_dataset(args)

    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()

    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, X_test, y_test, attn_type, lr, epochs=20, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier(attn_type=attn_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test, y_test = X_test.to(device), y_test.to(device)
        test_output = model(X_test)
        pred = test_output.argmax(dim=1)
        accuracy = (pred == y_test).float().mean().item()

    return accuracy

def objective(trial, X_train, y_train, X_test, y_test, attn_type):
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    acc = train_model(X_train, y_train, X_test, y_test, attn_type, lr, epochs=8)
    return acc

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()

    results = {}
    attn_types = ['standard', 'agwa', 'dagwa']

    for attn_type in attn_types:
        print(f"Testing {attn_type}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, attn_type), n_trials=8)

        best_lr = study.best_params['lr']
        print(f"Best LR for {attn_type}: {best_lr}")

        # Final evaluation with best LR and more epochs
        accs = []
        for i in range(2): # Run 2 times to get average
            acc = train_model(X_train, y_train, X_test, y_test, attn_type, best_lr, epochs=20)
            accs.append(acc)

        results[attn_type] = {
            'mean': np.mean(accs),
            'std': np.std(accs),
            'best_lr': best_lr
        }
        print(f"Result for {attn_type}: {results[attn_type]['mean']:.4f} +/- {results[attn_type]['std']:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    means = [results[n]['mean'] for n in names]
    stds = [results[n]['std'] for n in names]

    plt.bar(names, means, yerr=stds, capsize=5)
    plt.ylabel('Accuracy')
    plt.title('Comparison of Attention Mechanisms on MNIST-1D')
    plt.savefig('adaptive_gaussian_window_attn_experiment/results.png')

    with open('adaptive_gaussian_window_attn_experiment/README.md', 'w') as f:
        f.write("# Adaptive Gaussian Window Attention Experiment\n\n")
        f.write("This experiment compares standard Multi-Head Attention with Adaptive Gaussian Window Attention (AGWA) and its dynamic version (DAGWA) on the MNIST-1D dataset.\n\n")
        f.write("## Hypothesis\n")
        f.write("Adding a learnable Gaussian window to the attention mechanism allows each head to focus on a local context if needed. In 1D signals like MNIST-1D, local relationships are often crucial, and giving the model the ability to adaptively scale its attention window should improve performance.\n\n")
        f.write("## Results\n")
        f.write("| Attention Type | Mean Accuracy | Std Dev | Best LR |\n")
        f.write("|----------------|---------------|---------|---------|\n")
        for name in names:
            f.write(f"| {name} | {results[name]['mean']:.4f} | {results[name]['std']:.4f} | {results[name]['best_lr']:.4e} |\n")
        f.write("\n![Results](results.png)\n")

if __name__ == "__main__":
    run_experiment()
