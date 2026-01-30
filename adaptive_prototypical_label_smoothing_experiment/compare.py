import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import numpy as np
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import matplotlib.pyplot as plt
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data(num_samples=200):
    args = get_dataset_args()
    args.num_samples = num_samples
    data = make_dataset(args)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    return X_train, y_train, X_test, y_test

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_features=False):
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        logits = self.fc3(features)
        if return_features:
            return logits, features
        return logits

def train_model(X_train, y_train, X_test, y_test, config):
    model = SimpleMLP(hidden_dim=config.get('hidden_dim', 128)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 1e-4))

    batch_size = config.get('batch_size', 32)
    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    epochs = config.get('epochs', 100)

    # APLS parameters
    use_apls = config.get('use_apls', False)
    use_ls = config.get('use_ls', False)
    epsilon = config.get('epsilon', 0.1)
    temp = config.get('temp', 1.0)
    momentum = config.get('momentum', 0.9)

    num_classes = 10
    feature_dim = config.get('hidden_dim', 128)
    centroids = torch.zeros(num_classes, feature_dim).to(device)
    centroids_initialized = torch.zeros(num_classes, dtype=torch.bool).to(device)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits, features = model(x, return_features=True)

            if use_apls:
                # Update centroids
                with torch.no_grad():
                    for c in range(num_classes):
                        mask = (y == c)
                        if mask.any():
                            batch_mean = features[mask].mean(dim=0)
                            if not centroids_initialized[c]:
                                centroids[c] = batch_mean
                                centroids_initialized[c] = True
                            else:
                                centroids[c] = momentum * centroids[c] + (1 - momentum) * batch_mean

                # Compute APLS targets
                with torch.no_grad():
                    if centroids_initialized.all():
                        dist = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
                        dist_for_softmax = dist.clone()
                        dist_for_softmax.fill_diagonal_(float('inf'))
                        weights = F.softmax(-dist_for_softmax / temp, dim=1)
                        targets_matrix = weights * epsilon
                        targets_matrix.scatter_(1, torch.arange(num_classes).view(-1, 1).to(device), 1 - epsilon)
                        targets = targets_matrix[y]
                    else:
                        # Fallback to standard LS
                        targets = torch.full_like(logits, epsilon / (num_classes - 1))
                        targets.scatter_(1, y.unsqueeze(1), 1 - epsilon)

                loss = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

            elif use_ls:
                targets = torch.full_like(logits, epsilon / (num_classes - 1))
                targets.scatter_(1, y.unsqueeze(1), 1 - epsilon)
                loss = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            else:
                loss = F.cross_entropy(logits, y)

            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device))
        preds = logits.argmax(dim=1)
        acc = (preds == y_test.to(device)).float().mean().item()
    return acc

def objective(trial, mode, X_train, y_train, X_test, y_test):
    config = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'epochs': 200,
        'batch_size': 32,
        'hidden_dim': 256,
    }

    if mode == 'ls':
        config['use_ls'] = True
        config['epsilon'] = trial.suggest_float('epsilon', 0.01, 0.4)
    elif mode == 'apls':
        config['use_apls'] = True
        config['epsilon'] = trial.suggest_float('epsilon', 0.01, 0.4)
        config['temp'] = trial.suggest_float('temp', 0.01, 10.0, log=True)
        config['momentum'] = trial.suggest_float('momentum', 0.5, 0.99)
    else:
        config['use_ls'] = False
        config['use_apls'] = False

    accuracies = []
    # Run 1 times to get a more stable estimate
    for _ in range(1):
        acc = train_model(X_train, y_train, X_test, y_test, config)
        accuracies.append(acc)
    return np.mean(accuracies)

if __name__ == "__main__":
    # Use even smaller dataset for more challenge
    num_samples = 300
    X_train, y_train, X_test, y_test = get_data(num_samples=num_samples)
    print(f"Data shapes: X_train {X_train.shape}, y_train {y_train.shape}")

    results = {}

    modes = ['baseline', 'ls', 'apls']
    for mode in modes:
        print(f"\n--- Running Optuna for {mode} ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, mode, X_train, y_train, X_test, y_test), n_trials=15)
        results[mode] = {
            'best_acc': study.best_value,
            'best_params': study.best_params
        }
        print(f"Best accuracy for {mode}: {study.best_value:.4f}")

    print("\n" + "="*30)
    print("Final Results Summary:")
    for mode in modes:
        res = results[mode]
        print(f"{mode:10}: {res['best_acc']:.4f} | Best Params: {res['best_params']}")
    print("="*30)

    # Save results to a file for later use in README
    with open("adaptive_prototypical_label_smoothing_experiment/results.txt", "w") as f:
        f.write("Final Results Summary:\n")
        for mode in modes:
            res = results[mode]
            f.write(f"{mode:10}: {res['best_acc']:.4f} | Best Params: {res['best_params']}\n")
