import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import optuna
from .muon_optimizer import MuonOptimizer
from .muon_hook_optimizer import MuonHookOptimizer

# Define a deeper neural network
class DeeperNet(nn.Module):
    def __init__(self):
        super(DeeperNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Generate a synthetic dataset
X = torch.randn(100, 1, 10, 10)
y = torch.randint(0, 2, (100,))

def train(optimizer_class, lr, beta=0.9):
    model = DeeperNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.parameters(), lr=lr, beta=beta)

    losses = []
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

def objective(trial, optimizer_class):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    losses = train(optimizer_class, lr=lr)
    return losses[-1]

if __name__ == "__main__":
    # --- Optuna Study for MuonOptimizer ---
    print("Tuning learning rate for MuonOptimizer...")
    study_muon = optuna.create_study(direction="minimize")
    study_muon.optimize(lambda trial: objective(trial, MuonOptimizer), n_trials=50)
    best_lr_muon = study_muon.best_trial.params["lr"]
    print(f"Best LR for MuonOptimizer: {best_lr_muon}")

    # --- Optuna Study for MuonHookOptimizer ---
    print("\nTuning learning rate for MuonHookOptimizer...")
    study_muon_hook = optuna.create_study(direction="minimize")
    study_muon_hook.optimize(lambda trial: objective(trial, MuonHookOptimizer), n_trials=50)
    best_lr_muon_hook = study_muon_hook.best_trial.params["lr"]
    print(f"Best LR for MuonHookOptimizer: {best_lr_muon_hook}")

    # --- Final Training & Plotting ---
    print("\nRunning final training with optimal learning rates...")
    muon_losses = train(MuonOptimizer, lr=best_lr_muon)
    muon_hook_losses = train(MuonHookOptimizer, lr=best_lr_muon_hook)

    print("Training complete.")
    print(f"Final loss with standard Muon (lr={best_lr_muon:.4f}): {muon_losses[-1]:.4f}")
    print(f"Final loss with hook-based Muon (lr={best_lr_muon_hook:.4f}): {muon_hook_losses[-1]:.4f}")

    # Plot the learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(muon_losses, label=f"Standard Muon (lr={best_lr_muon:.4f})")
    plt.plot(muon_hook_losses, label=f"Hook-based Muon (lr={best_lr_muon_hook:.4f})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Muon Optimizer Comparison with Tuned Learning Rates")
    plt.savefig("muon_hook_experiment/muon_comparison_tuned.png")
    print("Plot saved to muon_hook_experiment/muon_comparison_tuned.png")
