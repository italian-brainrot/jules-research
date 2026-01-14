import torch
import torch.nn as nn
import torch.optim as optim
import mnist1d
from light_dataloader import TensorDataLoader
import optuna
import matplotlib.pyplot as plt
import numpy as np
import os
from optimizer import AdaptiveNoiseAdam

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST1D dataset
args = mnist1d.data.get_dataset_args()
data = mnist1d.data.get_dataset(args)

X_train = torch.tensor(data['x'], dtype=torch.float32).to(device)
y_train = torch.tensor(data['y'], dtype=torch.long).to(device)
X_test = torch.tensor(data['x_test'], dtype=torch.float32).to(device)
y_test = torch.tensor(data['y_test'], dtype=torch.long).to(device)

train_loader = TensorDataLoader((X_train, y_train), batch_size=128)
test_loader = TensorDataLoader((X_test, y_test), batch_size=128)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

def train(model, optimizer, criterion, train_loader):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.data[0])
    accuracy = 100. * correct / len(test_loader.data[0])
    return test_loss, accuracy

def objective(trial, optimizer_name):
    model = MLP().to(device)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        base_optimizer = optim.Adam(model.parameters(), lr=lr)
        noise_level = trial.suggest_float('noise_level', 1e-4, 1.0, log=True)
        optimizer = AdaptiveNoiseAdam(base_optimizer, noise_level=noise_level)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(10): # Reduced epochs for faster tuning
        train(model, optimizer, criterion, train_loader)

    _, accuracy = evaluate(model, criterion, test_loader)
    return accuracy

def run_experiment():
    # Tune hyperparameters for Adam
    study_adam = optuna.create_study(direction='maximize')
    study_adam.optimize(lambda trial: objective(trial, 'Adam'), n_trials=20)
    best_lr_adam = study_adam.best_params['lr']

    # Tune hyperparameters for AdaptiveNoiseAdam
    study_anadam = optuna.create_study(direction='maximize')
    study_anadam.optimize(lambda trial: objective(trial, 'AdaptiveNoiseAdam'), n_trials=20)
    best_lr_anadam = study_anadam.best_params['lr']
    best_noise_level = study_anadam.best_params['noise_level']

    # Train final models with best hyperparameters
    model_adam = MLP().to(device)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=best_lr_adam)

    model_anadam = MLP().to(device)
    base_optimizer_anadam = optim.Adam(model_anadam.parameters(), lr=best_lr_anadam)
    optimizer_anadam = AdaptiveNoiseAdam(base_optimizer_anadam, noise_level=best_noise_level)

    criterion = nn.CrossEntropyLoss()

    adam_losses = []
    anadam_losses = []

    epochs = 50
    for epoch in range(epochs):
        train(model_adam, optimizer_adam, criterion, train_loader)
        loss, _ = evaluate(model_adam, criterion, test_loader)
        adam_losses.append(loss)

        train(model_anadam, optimizer_anadam, criterion, train_loader)
        loss, _ = evaluate(model_anadam, criterion, test_loader)
        anadam_losses.append(loss)

        print(f'Epoch {epoch+1}/{epochs} - Adam Loss: {adam_losses[-1]:.4f}, AdaptiveNoiseAdam Loss: {anadam_losses[-1]:.4f}')

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(adam_losses, label='Adam')
    plt.plot(anadam_losses, label='AdaptiveNoiseAdam')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Adam vs. AdaptiveNoiseAdam')
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'comparison.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    run_experiment()
