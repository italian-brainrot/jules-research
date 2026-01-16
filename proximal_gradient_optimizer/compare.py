import torch
import torch.nn as nn
import optuna
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset, get_dataset_args
from optimizer import ProximalOptimizer, soft_thresholding

# --- 1. Dataset and Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = get_dataset_args()
data = get_dataset(args)
X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)

# Normalize data
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

dl_train = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)
dl_test = TensorDataLoader((X_test, y_test), batch_size=1024)

def get_model():
    return nn.Sequential(
        nn.Linear(X_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

# --- 2. Training and Evaluation ---
def train_and_evaluate(optimizer, model, epochs=20):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for inputs, targets in dl_train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dl_test:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(y_test)

# --- 3. Optuna Objective ---
def objective(trial, optimizer_name):
    model = get_model()
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'ProximalAdam':
        optimizer = ProximalOptimizer(
            model.parameters(),
            base_optimizer=torch.optim.Adam,
            prox_fn=soft_thresholding,
            lr=lr
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    final_val_loss = train_and_evaluate(optimizer, model)
    return final_val_loss

# --- 4. Main Execution ---
if __name__ == '__main__':
    # --- Tune Adam ---
    study_adam = optuna.create_study(direction='minimize')
    study_adam.optimize(lambda trial: objective(trial, 'Adam'), n_trials=30)
    best_lr_adam = study_adam.best_params['lr']
    print(f"Best LR for Adam: {best_lr_adam:.6f}")

    # --- Tune ProximalAdam ---
    study_proximal = optuna.create_study(direction='minimize')
    study_proximal.optimize(lambda trial: objective(trial, 'ProximalAdam'), n_trials=30)
    best_lr_proximal = study_proximal.best_params['lr']
    print(f"Best LR for ProximalAdam: {best_lr_proximal:.6f}")

    # --- Final Comparison ---
    print("\n--- Final Comparison ---")
    # Adam
    model_adam = get_model()
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=best_lr_adam)
    final_loss_adam = train_and_evaluate(optimizer_adam, model_adam, epochs=50)
    print(f"Final Validation Loss (Adam): {final_loss_adam:.4f}")

    # ProximalAdam
    model_proximal = get_model()
    optimizer_proximal = ProximalOptimizer(
        model_proximal.parameters(),
        base_optimizer=torch.optim.Adam,
        prox_fn=soft_thresholding,
        lr=best_lr_proximal
    )
    final_loss_proximal = train_and_evaluate(optimizer_proximal, model_proximal, epochs=50)
    print(f"Final Validation Loss (ProximalAdam): {final_loss_proximal:.4f}")

    if final_loss_proximal < final_loss_adam:
        print("\nConclusion: ProximalAdam performed better.")
    else:
        print("\nConclusion: Standard Adam performed better or equally.")
