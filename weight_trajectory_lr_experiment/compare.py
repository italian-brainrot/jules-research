import torch
import optuna
import plotly.graph_objects as go
from main import MLP, WeightTrajectoryLR, train_epoch, evaluate
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset, get_dataset_args
import torch.nn as nn

# 1. Objective Function for Optuna
def objective(trial, use_wrapper, train_loader, test_loader):
    input_size = 40
    hidden_size = 256
    num_classes = 10
    num_epochs = 20

    torch.manual_seed(42) # for reproducibility
    model = MLP(input_size, hidden_size, num_classes)

    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    # Create parameter groups for layer-wise LR
    param_groups = [
        {'params': model.fc1.parameters(), 'lr': lr},
        {'params': model.fc2.parameters(), 'lr': lr}
    ]
    base_optimizer = torch.optim.Adam(param_groups, lr=lr)

    if use_wrapper:
        beta = trial.suggest_float('beta', 0.1, 10.0, log=True)
        optimizer = WeightTrajectoryLR(base_optimizer, beta=beta)
    else:
        optimizer = base_optimizer

    criterion = nn.CrossEntropyLoss()

    min_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, optimizer, criterion)
        val_loss, _ = evaluate(model, test_loader, criterion)
        if val_loss < min_val_loss:
            min_val_loss = val_loss

    return min_val_loss

# 2. Main Comparison Logic
if __name__ == '__main__':
    # Load data once
    args = get_dataset_args()
    data = get_dataset(args, path='./mnist1d_data.pkl')
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.int64)
    X_test, y_test = torch.tensor(data['x_test'], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.int64)
    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

    # --- Baseline Study ---
    print("Running baseline Adam study...")
    study_baseline = optuna.create_study(direction='minimize')
    study_baseline.optimize(lambda trial: objective(trial, False, train_loader, test_loader), n_trials=30)
    print(f"Best baseline loss: {study_baseline.best_value:.4f}")
    print(f"Best baseline params: {study_baseline.best_params}")

    # --- Wrapped Optimizer Study ---
    print("\nRunning WeightTrajectoryLR study...")
    study_wrapped = optuna.create_study(direction='minimize')
    study_wrapped.optimize(lambda trial: objective(trial, True, train_loader, test_loader), n_trials=30)
    print(f"Best wrapped optimizer loss: {study_wrapped.best_value:.4f}")
    print(f"Best wrapped optimizer params: {study_wrapped.best_params}")

    # --- Plotting Results ---
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=[t.value for t in study_baseline.trials],
        name='Baseline Adam'
    ))
    fig.add_trace(go.Box(
        y=[t.value for t in study_wrapped.trials],
        name='WeightTrajectoryLR Adam'
    ))
    fig.update_layout(
        title='Comparison of Optimizer Performance',
        yaxis_title='Best Validation Loss',
        boxmode='group'
    )
    fig.write_image("comparison.png")
    print("\nGenerated comparison.png")
