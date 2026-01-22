import torch
import torch.nn as nn
import torch.nn as nn
import optuna
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset
import os
import plotly.graph_objects as go
from pickle import UnpicklingError

from main import SoftWeightProjection

# --- Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS = 20
N_EPOCHS = 20
BATCH_SIZE = 128
DATA_PATH = './mnist1d_data.pkl'

# --- Model Definition ---
class SimpleMLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, num_classes=10):
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

# --- Data Loading ---
def get_data_loaders():
    if not os.path.exists(DATA_PATH):
        print("Downloading dataset...")
        # In a restricted env, this might fail, but let's try
        from mnist1d.data import make_dataset, get_dataset_args
        defaults = get_dataset_args()
        defaults.num_samples = 10000
        data = make_dataset(defaults)
        torch.save(data, DATA_PATH)

    try:
        data = torch.load(DATA_PATH, weights_only=False)
    except UnpicklingError:
        print("Dataset file corrupted. Deleting and regenerating.")
        os.remove(DATA_PATH)
        return get_data_loaders()


    X_train = torch.tensor(data['x'], dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(data['y'], dtype=torch.long).to(DEVICE)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32).to(DEVICE)
    y_test = torch.tensor(data['y_test'], dtype=torch.long).to(DEVICE)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = TensorDataLoader((X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

# --- Training and Evaluation ---
def train_and_evaluate(model, optimizer, train_loader, val_loader, epochs):
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
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

    return val_losses

# --- Optuna Objective Functions ---
def objective_adam(trial, train_loader, val_loader):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    model = SimpleMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_losses = train_and_evaluate(model, optimizer, train_loader, val_loader, N_EPOCHS)

    return min(val_losses)

def objective_swp(trial, train_loader, val_loader):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    projection_strength = trial.suggest_float('projection_strength', 0.0, 0.9)

    model = SimpleMLP().to(DEVICE)
    base_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = SoftWeightProjection(base_optimizer, projection_strength=projection_strength)

    val_losses = train_and_evaluate(model, optimizer, train_loader, val_loader, N_EPOCHS)

    return min(val_losses)

# --- Main Execution ---
if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders()

    # --- Run Optuna Studies ---
    print("--- Tuning Adam ---")
    study_adam = optuna.create_study(direction='minimize')
    study_adam.optimize(lambda trial: objective_adam(trial, train_loader, val_loader), n_trials=N_TRIALS)

    print("\n--- Tuning SoftWeightProjection ---")
    study_swp = optuna.create_study(direction='minimize')
    study_swp.optimize(lambda trial: objective_swp(trial, train_loader, val_loader), n_trials=N_TRIALS)

    print("\n--- Results ---")
    print(f"Best Adam val_loss: {study_adam.best_value:.4f}")
    print(f"Best Adam params: {study_adam.best_params}")
    print(f"Best SWP val_loss: {study_swp.best_value:.4f}")
    print(f"Best SWP params: {study_swp.best_params}")

    # --- Train Best Models and Plot ---
    torch.manual_seed(42)
    adam_model = SimpleMLP().to(DEVICE)
    adam_optimizer = torch.optim.Adam(adam_model.parameters(), **study_adam.best_params)
    adam_losses = train_and_evaluate(adam_model, adam_optimizer, train_loader, val_loader, N_EPOCHS)

    torch.manual_seed(42)
    swp_model = SimpleMLP().to(DEVICE)
    swp_base_optimizer = torch.optim.Adam(swp_model.parameters(), lr=study_swp.best_params['lr'])
    swp_optimizer = SoftWeightProjection(swp_base_optimizer, projection_strength=study_swp.best_params['projection_strength'])
    swp_losses = train_and_evaluate(swp_model, swp_optimizer, train_loader, val_loader, N_EPOCHS)

    # --- Plotting ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=adam_losses, mode='lines', name=f'Adam (Best Val Loss: {min(adam_losses):.4f})'))
    fig.add_trace(go.Scatter(y=swp_losses, mode='lines', name=f'SWP (Best Val Loss: {min(swp_losses):.4f})'))
    fig.update_layout(
        title='Adam vs. Soft Weight Projection',
        xaxis_title='Epoch',
        yaxis_title='Validation Loss',
        legend_title='Optimizer'
    )
    fig.write_image("comparison.png")
    fig.write_html("comparison.html")
    print("\nSaved comparison plot to comparison.png and comparison.html")
