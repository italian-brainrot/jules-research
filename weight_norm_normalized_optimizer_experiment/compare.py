import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import matplotlib.pyplot as plt
import os
from optimizer import WNGD

# --- 1. Dataset and Dataloader Setup ---
def get_dataloaders():
    defaults = get_dataset_args()
    defaults.num_samples = 4000
    data = make_dataset(defaults)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)
    return train_loader, test_loader

# --- 2. Model Definition ---
class SimpleMLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# --- 3. Training and Evaluation Logic ---
def train_and_evaluate(model, optimizer, train_loader, test_loader, epochs=50):
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)
        history.append(avg_loss)

    return min(history), history


# --- 4. Optuna Objective Function ---
def objective(trial, optimizer_name, train_loader, test_loader):
    torch.manual_seed(42) # for reproducibility
    model = SimpleMLP()

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "WNGD":
        optimizer = WNGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Unknown optimizer")

    min_val_loss, _ = train_and_evaluate(model, optimizer, train_loader, test_loader, epochs=25) # Fewer epochs for tuning
    return min_val_loss

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    optimizers_to_compare = ["Adam", "WNGD"]
    best_params = {}
    histories = {}

    # --- Optuna Hyperparameter Tuning ---
    print("--- Starting Hyperparameter Tuning ---")
    for optimizer_name in optimizers_to_compare:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, optimizer_name, train_loader, test_loader), n_trials=15, timeout=300)
        best_params[optimizer_name] = study.best_trial.params
        print(f"Best params for {optimizer_name}: {study.best_trial.params} (loss: {study.best_trial.value:.4f})")

    # --- Final Training with Best Hyperparameters ---
    print("\n--- Starting Final Training ---")
    for optimizer_name, params in best_params.items():
        print(f"Training {optimizer_name} with lr={params['lr']:.6f}...")
        torch.manual_seed(42) # Re-seed for fair comparison
        model = SimpleMLP()

        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        else: # WNGD
            optimizer = WNGD(model.parameters(), lr=params['lr'])

        min_loss, history = train_and_evaluate(model, optimizer, train_loader, test_loader, epochs=100)
        histories[optimizer_name] = history
        print(f"Final minimum validation loss for {optimizer_name}: {min_loss:.4f}")

    # --- Plotting Results ---
    plt.figure(figsize=(10, 6))
    for optimizer_name, history in histories.items():
        plt.plot(history, label=f"{optimizer_name} (Best LR: {best_params[optimizer_name]['lr']:.4f})")

    plt.title("Optimizer Comparison: Adam vs. WNGD on mnist1d")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, max(max(h) for h in histories.values()) * 1.1) # Set ylim for better viewing

    # Save the plot to the experiment's directory
    plot_path = os.path.join(os.path.dirname(__file__), "optimizer_comparison.png")
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")
