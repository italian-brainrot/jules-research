import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import numpy as np
import matplotlib.pyplot as plt
import os
import json

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

class IterativeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_steps, variant='baseline'):
        super().__init__()
        self.num_steps = num_steps
        self.variant = variant
        self.hidden_dim = hidden_dim

        self.encoder = nn.Linear(input_dim, hidden_dim)

        refiner_input_dim = hidden_dim + input_dim
        if variant == 'feedback':
            refiner_input_dim += num_classes + 1 # + logits + loss_pred

        self.refiner = nn.Sequential(
            nn.Linear(refiner_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.head = nn.Linear(hidden_dim, num_classes)

        if variant in ['auxiliary', 'feedback']:
            self.loss_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(self, x):
        h = self.encoder(x)
        logits_list = []
        loss_preds_list = []

        curr_logits = torch.zeros(x.size(0), 10, device=x.device)
        curr_loss_pred = torch.zeros(x.size(0), 1, device=x.device)

        for _ in range(self.num_steps):
            if self.variant == 'feedback':
                refiner_input = torch.cat([h, x, curr_logits, curr_loss_pred], dim=-1)
            else:
                refiner_input = torch.cat([h, x], dim=-1)

            h = h + self.refiner(refiner_input)

            curr_logits = self.head(h)
            logits_list.append(curr_logits)

            if self.variant in ['auxiliary', 'feedback']:
                curr_loss_pred = self.loss_predictor(h)
                loss_preds_list.append(curr_loss_pred)

        return logits_list, loss_preds_list

def train_eval(config, X_train, y_train, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=config.get('batch_size', 1024), shuffle=True)

    model = IterativeModel(
        input_dim=40,
        hidden_dim=128,
        num_classes=10,
        num_steps=config['num_steps'],
        variant=config['variant']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config['epochs']):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits_list, loss_preds_list = model(batch_x)

            total_loss = 0
            for t in range(len(logits_list)):
                ce_loss_none = F.cross_entropy(logits_list[t], batch_y, reduction='none')
                ce_loss = ce_loss_none.mean()
                total_loss += ce_loss

                if config['variant'] in ['auxiliary', 'feedback']:
                    mse_loss = F.mse_loss(loss_preds_list[t].squeeze(), ce_loss_none.detach())
                    total_loss += config['lambda_loss'] * mse_loss

            total_loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits_list, _ = model(X_test)
        accs = []
        for t in range(len(test_logits_list)):
            preds = test_logits_list[t].argmax(dim=-1)
            accuracy = (preds == y_test).float().mean().item()
            accs.append(accuracy)

    return accs

def objective(trial, variant, X_train, y_train, X_test, y_test):
    config = {
        'variant': variant,
        'num_steps': 5,
        'hidden_dim': 128,
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'epochs': 10,
        'batch_size': 1024,
        'lambda_loss': trial.suggest_float('lambda_loss', 0.01, 10.0, log=True) if variant != 'baseline' else 0
    }
    accs = train_eval(config, X_train, y_train, X_test, y_test)
    return accs[-1] # Return final step accuracy

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    results = {}

    for variant in ['baseline', 'auxiliary', 'feedback']:
        print(f"Optimizing {variant}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, variant, X_train, y_train, X_test, y_test), n_trials=10)

        best_config = {
            'variant': variant,
            'num_steps': 5,
            'hidden_dim': 128,
            'lr': study.best_params['lr'],
            'epochs': 20,
            'batch_size': 1024,
            'lambda_loss': study.best_params.get('lambda_loss', 0)
        }

        print(f"Best params for {variant}: {study.best_params}")

        # Final evaluation with best params and more epochs
        accs = train_eval(best_config, X_train, y_train, X_test, y_test)
        results[variant] = {
            'best_params': study.best_params,
            'test_accuracies': accs
        }

    with open('self_assessed_iterative_refinement/results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Plot results
    plt.figure(figsize=(10, 6))
    for variant, res in results.items():
        plt.plot(range(1, 6), res['test_accuracies'], label=variant, marker='o')
    plt.xlabel('Iteration Step')
    plt.ylabel('Test Accuracy')
    plt.title('Iterative Refinement Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('self_assessed_iterative_refinement/accuracy_plot.png')
