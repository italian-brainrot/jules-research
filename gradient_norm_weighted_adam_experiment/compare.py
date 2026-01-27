import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import plotly.graph_objects as go
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
from optimizer import GNWAdam
import os

# Set a fixed seed for reproducibility
torch.manual_seed(42)

# Load MNIST1D dataset
args = get_dataset_args()
data = make_dataset(args)
X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
X_val, y_val = torch.tensor(data['x_test'], dtype=torch.float32), torch.tensor(data['y_test'])

# Create TensorDataLoaders
train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
val_loader = TensorDataLoader((X_val, y_val), batch_size=128)

# Define the neural network model
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = X_train.shape[1]
hidden_size = 64
num_classes = len(torch.unique(y_train))

def train(model, optimizer, train_loader, val_loader, epochs=50):
    criterion = nn.CrossEntropyLoss()
    val_losses = []
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        val_losses.append(val_loss / len(val_loader))
    return val_losses

def objective(trial, optimizer_class):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    model = SimpleMLP(input_size, hidden_size, num_classes)
    optimizer = optimizer_class(model.parameters(), lr=lr)

    val_losses = train(model, optimizer, train_loader, val_loader, epochs=25) # Shorter epochs for tuning
    return min(val_losses)

# Run Optuna studies
study_adam = optuna.create_study(direction='minimize')
study_adam.optimize(lambda trial: objective(trial, optim.Adam), n_trials=10)

study_gnw_adam = optuna.create_study(direction='minimize')
study_gnw_adam.optimize(lambda trial: objective(trial, GNWAdam), n_trials=10)

# Get best hyperparameters
best_lr_adam = study_adam.best_params['lr']
best_lr_gnw_adam = study_gnw_adam.best_params['lr']

# Train final models with best hyperparameters
model_adam = SimpleMLP(input_size, hidden_size, num_classes)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=best_lr_adam)
history_adam = train(model_adam, optimizer_adam, train_loader, val_loader, epochs=50)

model_gnw_adam = SimpleMLP(input_size, hidden_size, num_classes)
optimizer_gnw_adam = GNWAdam(model_gnw_adam.parameters(), lr=best_lr_gnw_adam)
history_gnw_adam = train(model_gnw_adam, optimizer_gnw_adam, train_loader, val_loader, epochs=50)

# Plotting the results
fig = go.Figure()
fig.add_trace(go.Scatter(y=history_adam, mode='lines', name=f'Adam (Best LR: {best_lr_adam:.5f})'))
fig.add_trace(go.Scatter(y=history_gnw_adam, mode='lines', name=f'GNW-Adam (Best LR: {best_lr_gnw_adam:.5f})'))
fig.update_layout(
    title='Adam vs. GNW-Adam Validation Loss',
    xaxis_title='Epoch',
    yaxis_title='Validation Loss',
    legend_title='Optimizer',
    template='plotly_white'
)

# Save plot
output_dir = os.path.dirname(__file__)
fig.write_html(os.path.join(output_dir, "comparison.html"))
print(f"Comparison plot saved to {os.path.join(output_dir, 'comparison.html')}")

# Print final results
print(f"Final validation loss for Adam: {min(history_adam)}")
print(f"Final validation loss for GNW-Adam: {min(history_gnw_adam)}")
