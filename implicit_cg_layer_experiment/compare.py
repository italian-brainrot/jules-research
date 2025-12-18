import torch
import torch.nn as nn
import optuna
import copy
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import matplotlib.pyplot as plt
import os

# --- Model Definitions ---

def batched_cg(A, b, max_iter=10):
    x = torch.zeros_like(b)
    r = b - torch.einsum('ij,bj->bi', A, x)
    p = r
    rs_old = torch.sum(r * r, dim=-1, keepdim=True)

    for _ in range(max_iter):
        Ap = torch.einsum('ij,bj->bi', A, p)
        alpha = rs_old / (torch.sum(p * Ap, dim=-1, keepdim=True) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r, dim=-1, keepdim=True)
        if torch.all(rs_new < 1e-6):
            break
        p = r + (rs_new / (rs_old + 1e-8)) * p
        rs_old = rs_new
    return x

class ImplicitCGLayer(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        weight = torch.randn(features, features)
        self.weight = nn.Parameter(weight @ weight.t())

    def forward(self, x):
        A = self.weight.t() @ self.weight + torch.eye(self.features, device=x.device) * 1e-3
        return batched_cg(A, x)

class ImplicitCGMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.cg_layer = ImplicitCGLayer(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.cg_layer(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class BaselineMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# --- Utility Functions ---

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 4000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)
    return train_loader, test_loader, data['x'].shape[1], len(data['templates']['y'])

def train(model, train_loader, test_loader, lr, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        history['train_loss'].append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        history['val_loss'].append(val_loss / len(test_loader))
        history['val_acc'].append(correct / total)
    return history

# --- Main Experiment ---

def objective(trial, model_class, train_loader, test_loader, input_size, num_classes):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    model = model_class(input_size, 128, num_classes)
    history = train(model, train_loader, test_loader, lr, epochs=10)
    return min(history['val_loss'])

def main():
    train_loader, test_loader, input_size, num_classes = get_data()

    study_baseline = optuna.create_study(direction='minimize')
    study_baseline.optimize(lambda trial: objective(trial, BaselineMLP, train_loader, test_loader, input_size, num_classes), n_trials=20)
    best_lr_baseline = study_baseline.best_params['lr']

    study_cg = optuna.create_study(direction='minimize')
    study_cg.optimize(lambda trial: objective(trial, ImplicitCGMLP, train_loader, test_loader, input_size, num_classes), n_trials=20)
    best_lr_cg = study_cg.best_params['lr']

    torch.manual_seed(42)
    baseline_model = BaselineMLP(input_size, 128, num_classes)
    cg_model = ImplicitCGMLP(input_size, 128, num_classes)

    cg_model.fc1.weight.data = copy.deepcopy(baseline_model.fc1.weight.data)
    cg_model.fc1.bias.data = copy.deepcopy(baseline_model.fc1.bias.data)
    cg_model.fc2.weight.data = copy.deepcopy(baseline_model.fc3.weight.data)
    cg_model.fc2.bias.data = copy.deepcopy(baseline_model.fc3.bias.data)

    baseline_history = train(baseline_model, train_loader, test_loader, best_lr_baseline, epochs=20)
    cg_history = train(cg_model, train_loader, test_loader, best_lr_cg, epochs=20)

    plt.figure(figsize=(10, 5))
    plt.plot(baseline_history['val_loss'], label='Baseline MLP')
    plt.plot(cg_history['val_loss'], label='Implicit CG MLP')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'comparison.png'))

if __name__ == '__main__':
    main()
