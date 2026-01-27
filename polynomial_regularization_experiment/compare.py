
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import os
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset_args, make_dataset
import plotly
import kaleido

# --- Dataset and Model ---

def get_data():
    """Fetches and prepares the mnist1d dataset."""
    args = get_dataset_args()
    data = make_dataset(args)

    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()

    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=512)
    return train_loader, test_loader

class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron."""
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
        return self.layers(x.view(x.size(0), -1))

# --- Training and Evaluation ---

def evaluate(model, loader):
    """Evaluates the model's accuracy on the given data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train_baseline(trial, train_loader, test_loader):
    """Trains a model with a standard Adam optimizer."""
    torch.manual_seed(42)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    min_val_loss = float('inf')

    for epoch in range(10):
        model.train()
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_acc = evaluate(model, test_loader)
        min_val_loss = min(min_val_loss, 1 - val_acc)

    return min_val_loss

def train_gradient_inertia(trial, train_loader, test_loader):
    """Trains a model with Gradient Inertia regularization."""
    torch.manual_seed(42)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    reg_lambda = trial.suggest_float('reg_lambda', 1e-5, 1e2, log=True)
    beta1 = 0.9 # Standard Adam beta1

    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = nn.CrossEntropyLoss()

    min_val_loss = float('inf')

    for epoch in range(10):
        model.train()
        for data, labels in train_loader:
            optimizer.zero_grad()

            # Standard loss
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Calculate gradient (g_t) w.r.t. the loss, creating a graph
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            inertia_reg = 0.0

            # Get previous momentum (m_{t-1}) from optimizer state
            for i, p in enumerate(model.parameters()):
                state = optimizer.state[p]
                if 'exp_avg' in state:
                    m_prev = state['exp_avg']
                    g_t = grads[i]
                    # Penalty is (1-beta1)^2 * ||g_t - m_{t-1}||^2
                    inertia_reg += ((1 - beta1)**2) * torch.sum((g_t - m_prev)**2)

            total_loss = loss + reg_lambda * inertia_reg

            # Backpropagate the total loss (including the gradient penalty)
            total_loss.backward()
            optimizer.step()

        val_acc = evaluate(model, test_loader)
        min_val_loss = min(min_val_loss, 1 - val_acc)

    return min_val_loss

# --- Main Execution ---

def main():
    """Runs the Optuna study to compare the two methods."""
    train_loader, test_loader = get_data()
    n_trials = 15 # Reduced trials to prevent timeout

    # --- Baseline Study ---
    study_baseline = optuna.create_study(direction='minimize')
    study_baseline.optimize(lambda trial: train_baseline(trial, train_loader, test_loader), n_trials=n_trials)

    print("Best baseline trial:")
    best_trial_baseline = study_baseline.best_trial
    print(f"  Value: {best_trial_baseline.value:.4f}")
    print(f"  Params: {best_trial_baseline.params}")

    # --- Gradient Inertia Study ---
    study_inertia = optuna.create_study(direction='minimize')
    study_inertia.optimize(lambda trial: train_gradient_inertia(trial, train_loader, test_loader), n_trials=n_trials)

    print("\nBest Gradient Inertia trial:")
    best_trial_inertia = study_inertia.best_trial
    print(f"  Value: {best_trial_inertia.value:.4f}")
    print(f"  Params: {best_trial_inertia.params}")

    # --- Visualization and Saving Results ---
    output_dir = os.path.dirname(__file__)
    try:
        # Combined plot
        fig = optuna.visualization.plot_optimization_history(
            [study_baseline, study_inertia],
        )
        # Manually set titles
        fig.data[0].name = 'Baseline Adam'
        fig.data[1].name = 'Gradient Inertia'
        fig.update_layout(title_text='Optimization History Comparison', title_x=0.5)

        # Save plot
        plot_path = os.path.join(output_dir, "comparison_optimization_history.png")
        fig.write_image(plot_path)

        print(f"\nPlots saved to '{plot_path}'")

    except (ImportError, ValueError) as e:
        print(f"\nCould not generate or save plots due to an error: {e}")
        print("Please ensure plotly and kaleido are installed: pip install plotly kaleido")


if __name__ == "__main__":
    main()
