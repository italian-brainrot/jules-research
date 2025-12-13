import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import matplotlib.pyplot as plt
import os
import copy

# Make the script runnable from the root directory
from cyclical_gradient_inversion.optimizer import CyclicalGradientInversion

# Define the model
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 5000
    defaults.train_split = 0.8
    data = make_dataset(defaults)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.int64)
    X_val = torch.tensor(data['x_test'], dtype=torch.float32)
    y_val = torch.tensor(data['y_test'], dtype=torch.int64)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    val_loader = TensorDataLoader((X_val, y_val), batch_size=128)

    return train_loader, val_loader

def train_and_evaluate(model, optimizer, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    val_losses = []
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
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        val_losses.append(avg_loss)

    return val_losses

def objective(trial, optimizer_name, initial_model_state, train_loader, val_loader):
    model = SimpleMLP()
    model.load_state_dict(initial_model_state)

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "CGI_Adam":
        inversion_frequency = trial.suggest_int("inversion_frequency", 2, 100)
        base_optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = CyclicalGradientInversion(base_optimizer, inversion_frequency=inversion_frequency)
    else:
        raise ValueError("Unknown optimizer")

    val_losses = train_and_evaluate(model, optimizer, train_loader, val_loader, epochs=10)

    return min(val_losses)

def main():
    train_loader, val_loader = get_data()

    # For fair comparison
    initial_model = SimpleMLP()
    initial_model_state = copy.deepcopy(initial_model.state_dict())

    # Tune Adam
    study_adam = optuna.create_study(direction="minimize")
    study_adam.optimize(lambda trial: objective(trial, "Adam", initial_model_state, train_loader, val_loader), n_trials=30)
    best_lr_adam = study_adam.best_params["lr"]
    print(f"Best LR for Adam: {best_lr_adam}")

    # Tune CGI(Adam)
    study_cgi = optuna.create_study(direction="minimize")
    study_cgi.optimize(lambda trial: objective(trial, "CGI_Adam", initial_model_state, train_loader, val_loader), n_trials=30)
    best_lr_cgi = study_cgi.best_params["lr"]
    best_freq_cgi = study_cgi.best_params["inversion_frequency"]
    print(f"Best LR for CGI(Adam): {best_lr_cgi}")
    print(f"Best Inversion Frequency for CGI(Adam): {best_freq_cgi}")

    # Final comparison run
    epochs = 50

    # Adam
    model_adam = SimpleMLP()
    model_adam.load_state_dict(initial_model_state)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=best_lr_adam)
    adam_losses = train_and_evaluate(model_adam, optimizer_adam, train_loader, val_loader, epochs=epochs)

    # CGI(Adam)
    model_cgi = SimpleMLP()
    model_cgi.load_state_dict(initial_model_state)
    base_optimizer_cgi = optim.Adam(model_cgi.parameters(), lr=best_lr_cgi)
    optimizer_cgi = CyclicalGradientInversion(base_optimizer_cgi, inversion_frequency=best_freq_cgi)
    cgi_losses = train_and_evaluate(model_cgi, optimizer_cgi, train_loader, val_loader, epochs=epochs)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(adam_losses, label="Adam")
    plt.plot(cgi_losses, label=f"CGI(Adam) - Freq: {best_freq_cgi}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Adam vs Cyclical Gradient Inversion (Adam)")
    plt.legend()
    plt.grid(True)

    # Save plot in the experiment directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, "comparison.png"))
    print(f"Plot saved to {os.path.join(script_dir, 'comparison.png')}")

if __name__ == "__main__":
    main()
