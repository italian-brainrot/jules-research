import torch
import torch.nn as nn
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import matplotlib.pyplot as plt
import copy
import os
import optuna

from .optimizer import MetaCurvatureLR

def get_model(input_dim=40, num_classes=10):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes)
    )

def train(model, optimizer, dl_train, dl_test, n_epochs=10):
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_train_loss = 0
        for inputs, targets in dl_train:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            def closure():
                return loss

            if isinstance(optimizer, MetaCurvatureLR):
                loss.backward(create_graph=True)
                optimizer.step(closure)
            else:
                loss.backward()
                optimizer.step()

            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(dl_train))

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in dl_test:
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
        test_losses.append(test_loss / len(dl_test))
        # print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

    return train_losses, test_losses

def objective(trial, optimizer_name, base_model, dl_train, dl_test):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    model = copy.deepcopy(base_model)

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'MetaCurvatureLR':
        base_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = MetaCurvatureLR(base_optimizer, update_freq=10, alpha=1.0)
    else:
        raise ValueError("Unknown optimizer")

    # Use fewer epochs for tuning to speed up the process
    _, test_losses = train(model, optimizer, dl_train, dl_test, n_epochs=5)

    return test_losses[-1]

def main():
    # --- Data Loading ---
    defaults = get_dataset_args()
    defaults.num_samples = 5000 # Using a smaller dataset for faster tuning
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])

    dl_train = TensorDataLoader((X_train, y_train), batch_size=32, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=32, shuffle=False)

    # --- Model Initialization ---
    base_model = get_model()

    # --- Hyperparameter Tuning ---
    print("Tuning learning rate for Adam...")
    study_adam = optuna.create_study(direction='minimize')
    study_adam.optimize(lambda trial: objective(trial, 'Adam', base_model, dl_train, dl_test), n_trials=20)
    best_lr_adam = study_adam.best_params['lr']
    print(f"Best LR for Adam: {best_lr_adam}")

    print("\nTuning learning rate for MetaCurvatureLR(Adam)...")
    study_meta = optuna.create_study(direction='minimize')
    study_meta.optimize(lambda trial: objective(trial, 'MetaCurvatureLR', base_model, dl_train, dl_test), n_trials=20)
    best_lr_meta = study_meta.best_params['lr']
    print(f"Best LR for MetaCurvatureLR(Adam): {best_lr_meta}")

    # --- Baseline: Adam with tuned LR ---
    print("\nTraining with tuned Adam...")
    model_adam = copy.deepcopy(base_model)
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=best_lr_adam)
    train_losses_adam, test_losses_adam = train(model_adam, optimizer_adam, dl_train, dl_test, n_epochs=15)

    # --- Experiment: MetaCurvatureLR(Adam) with tuned LR ---
    print("\nTraining with tuned MetaCurvatureLR(Adam)...")
    model_meta = copy.deepcopy(base_model)
    base_optimizer_meta = torch.optim.Adam(model_meta.parameters(), lr=best_lr_meta)
    optimizer_meta = MetaCurvatureLR(base_optimizer_meta, update_freq=10, alpha=1.0)
    train_losses_meta, test_losses_meta = train(model_meta, optimizer_meta, dl_train, dl_test, n_epochs=15)

    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_adam, label=f'Adam (LR={best_lr_adam:.1e})')
    plt.plot(train_losses_meta, label=f'MetaCurvatureLR(Adam) (LR={best_lr_meta:.1e})')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(test_losses_adam, label=f'Adam (LR={best_lr_adam:.1e})')
    plt.plot(test_losses_meta, label=f'MetaCurvatureLR(Adam) (LR={best_lr_meta:.1e})')
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save the plot in the experiment's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'loss_comparison_tuned.png'))
    # plt.show()
    print("\nSaved plot to loss_comparison_tuned.png")

if __name__ == "__main__":
    main()
