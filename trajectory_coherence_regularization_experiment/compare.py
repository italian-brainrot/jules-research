import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import optuna
import torch.nn.functional as F

# Dataset
def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 4000
    defaults.test_samples = 1000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()

    # Cast to float32
    if X_train.dtype != torch.float32:
        X_train = X_train.float()
    if X_test.dtype != torch.float32:
        X_test = X_test.float()

    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)
    return train_loader, test_loader

# Model
class SimpleMLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def get_flat_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])

# Optuna Objective for Baseline
def objective_baseline(trial):
    torch.manual_seed(42)
    train_loader, test_loader = get_data()
    model = SimpleMLP()

    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            val_loss += criterion(output, target).item()

    return val_loss / len(test_loader)

# Optuna Objective for Regularized Model
def objective_regularized(trial):
    torch.manual_seed(42)
    train_loader, test_loader = get_data()
    model = SimpleMLP()

    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    lam = trial.suggest_float('lam', 1e-5, 1e-1, log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    w_history = [get_flat_params(model).detach().clone()]

    # Training loop
    for epoch in range(10):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            reg_loss = torch.tensor(0.0)
            if len(w_history) >= 3:
                # w_history contains [w_t-3, w_t-2, w_t-1]
                delta_t_minus_1 = w_history[-1] - w_history[-2]
                delta_t_minus_2 = w_history[-2] - w_history[-3]

                if delta_t_minus_1.norm() > 1e-8 and delta_t_minus_2.norm() > 1e-8:
                    cos = F.cosine_similarity(delta_t_minus_1, delta_t_minus_2, dim=0)
                    reg_loss = lam * F.relu(-cos)

            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()

            w_history.append(get_flat_params(model).detach().clone())
            if len(w_history) > 3:
                w_history.pop(0)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            val_loss += criterion(output, target).item()

    return val_loss / len(test_loader)


if __name__ == '__main__':
    print("Running baseline study...")
    study_baseline = optuna.create_study(direction='minimize')
    study_baseline.optimize(objective_baseline, n_trials=10)

    print("\nRunning regularized study...")
    study_regularized = optuna.create_study(direction='minimize')
    study_regularized.optimize(objective_regularized, n_trials=10)

    print("\n\n--- Results ---")
    print("Best baseline trial:")
    baseline_trial = study_baseline.best_trial
    print(f"  Value: {baseline_trial.value}")
    print("  Params: ")
    for key, value in baseline_trial.params.items():
        print(f"    {key}: {value}")

    print("\nBest regularized trial:")
    reg_trial = study_regularized.best_trial
    print(f"  Value: {reg_trial.value}")
    print("  Params: ")
    for key, value in reg_trial.params.items():
        print(f"    {key}: {value}")
