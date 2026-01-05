import torch
import torch.nn as nn
import optuna
import matplotlib.pyplot as plt
import os
import numpy as np
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args

# Make sure the optimizer is importable
from optimizer import RelativeGradientPredictiveCoding

# --- Configuration ---
DEVICE = torch.device("cpu")
N_TRIALS = 20
N_EPOCHS = 10
BATCH_SIZE = 128
INPUT_DIM = 40
OUTPUT_DIM = 10

# --- Reproducibility ---
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- Data Loading ---
def get_data():
    args = get_dataset_args()
    args.num_samples = 5000
    data = make_dataset(args)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.int64)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.int64)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=BATCH_SIZE)

    return train_loader, test_loader

# --- Model Definition ---
def get_model():
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, OUTPUT_DIM)
    ).to(DEVICE)

# --- Training and Evaluation ---
def train_eval(model, optimizer, train_loader, test_loader, criterion):
    history = []
    for epoch in range(N_EPOCHS):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(x_batch)
                total_loss += criterion(outputs, y_batch).item()

        avg_loss = total_loss / len(test_loader)
        history.append(avg_loss)
    return history

# --- Optuna Objective ---
def objective(trial, optimizer_name, train_loader, test_loader, initial_model_state):
    model = get_model()
    model.load_state_dict(initial_model_state)
    criterion = nn.CrossEntropyLoss()

    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RGPC-Adam':
        base_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = RelativeGradientPredictiveCoding(model.parameters(), base_optimizer)
    else:
        raise ValueError("Unknown optimizer")

    history = train_eval(model, optimizer, train_loader, test_loader, criterion)

    trial.set_user_attr("history", history)
    return min(history) # Return the best validation loss

# --- Main Execution ---
if __name__ == "__main__":
    train_loader, test_loader = get_data()

    initial_model = get_model()
    initial_model_state = initial_model.state_dict()

    # --- Tune Adam ---
    print("Tuning Adam...")
    study_adam = optuna.create_study(direction='minimize')
    study_adam.optimize(lambda trial: objective(trial, 'Adam', train_loader, test_loader, initial_model_state), n_trials=N_TRIALS)
    best_trial_adam = study_adam.best_trial

    # --- Tune RGPC-Adam ---
    print("Tuning RGPC-Adam...")
    study_rgpc = optuna.create_study(direction='minimize')
    study_rgpc.optimize(lambda trial: objective(trial, 'RGPC-Adam', train_loader, test_loader, initial_model_state), n_trials=N_TRIALS)
    best_trial_rgpc = study_rgpc.best_trial

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(best_trial_adam.user_attrs['history'], label=f'Adam (Best LR: {best_trial_adam.params["lr"]:.4f})')
    plt.plot(best_trial_rgpc.user_attrs['history'], label=f'RGPC-Adam (Best LR: {best_trial_rgpc.params["lr"]:.4f})')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'comparison.png'))

    print("\n--- Results ---")
    print(f"Best Adam Validation Loss: {best_trial_adam.value:.4f}")
    print(f"Best RGPC-Adam Validation Loss: {best_trial_rgpc.value:.4f}")
    print(f"Plot saved to {os.path.join(script_dir, 'comparison.png')}")
