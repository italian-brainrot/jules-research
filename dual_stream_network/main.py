import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import matplotlib.pyplot as plt

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DualStreamNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DualStreamNetwork, self).__init__()
        # Stream 1: Raw data
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Stream 2: Fourier transform
        self.mlp2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # FFT of real signal is symmetric, but we'll use the full magnitude spectrum
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Combiner
        self.combiner = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # Stream 1
        out1 = self.mlp1(x)

        # Stream 2
        # Apply FFT. x is a batch of 1D signals.
        x_fft = torch.fft.fft(x, dim=1)
        x_fft_mag = torch.abs(x_fft)
        out2 = self.mlp2(x_fft_mag)

        # Combine
        combined = torch.cat((out1, out2), dim=1)
        output = self.combiner(combined)
        return output

def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def tune_learning_rate(model_class, model_args, train_loader, val_loader, criterion, lrs_to_try, epochs):
    """
    Tunes the learning rate for a given model by training it for a few epochs for each learning rate
    and returning the one that yields the lowest validation loss.
    """
    best_lr = None
    best_val_loss = float('inf')

    print(f"--- Tuning LR for {model_class.__name__} ---")
    for lr in lrs_to_try:
        # Re-initialize the model and optimizer for each LR to ensure a fair comparison
        model = model_class(**model_args)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train for a set number of epochs
        for epoch in range(epochs):
            train_model(model, train_loader, optimizer, criterion)

        # Evaluate the model
        val_loss = evaluate_model(model, val_loader, criterion)
        print(f"LR: {lr} | Final Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lr = lr

    print(f"Best LR for {model_class.__name__}: {best_lr} (Val Loss: {best_val_loss:.4f})\n")
    return best_lr

def main():
    # --- Data Loading ---
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])

    dl_train = TensorDataLoader((X_train, y_train), batch_size=32, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=32, shuffle=False)

    # --- Hyperparameters ---
    input_dim = 40
    output_dim = 10
    baseline_hidden_dim = 64
    dual_stream_hidden_dim = 40
    lrs_to_try = [0.01, 0.001, 0.0001]
    epochs = 20
    tuning_epochs = 5 # Use fewer epochs for tuning to speed up the process

    # --- LR Tuning Phase ---
    criterion = nn.CrossEntropyLoss()

    baseline_model_args = {'input_dim': input_dim, 'hidden_dim': baseline_hidden_dim, 'output_dim': output_dim}
    best_lr_baseline = tune_learning_rate(BaselineMLP, baseline_model_args, dl_train, dl_test, criterion, lrs_to_try, tuning_epochs)

    dual_stream_model_args = {'input_dim': input_dim, 'hidden_dim': dual_stream_hidden_dim, 'output_dim': output_dim}
    best_lr_dual_stream = tune_learning_rate(DualStreamNetwork, dual_stream_model_args, dl_train, dl_test, criterion, lrs_to_try, tuning_epochs)

    # --- Final Training Phase ---
    print("--- Starting Final Training with Best LRs ---")
    baseline_model = BaselineMLP(**baseline_model_args)
    dual_stream_model = DualStreamNetwork(**dual_stream_model_args)

    # --- Parameter Count ---
    baseline_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    dual_stream_params = sum(p.numel() for p in dual_stream_model.parameters() if p.requires_grad)
    print(f"Baseline MLP parameters: {baseline_params}")
    print(f"Dual-Stream Network parameters: {dual_stream_params}")

    # --- Training ---
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=best_lr_baseline)
    dual_stream_optimizer = optim.Adam(dual_stream_model.parameters(), lr=best_lr_dual_stream)

    baseline_train_losses, baseline_val_losses = [], []
    dual_stream_train_losses, dual_stream_val_losses = [], []

    for epoch in range(epochs):
        # Baseline
        train_loss = train_model(baseline_model, dl_train, baseline_optimizer, criterion)
        val_loss = evaluate_model(baseline_model, dl_test, criterion)
        baseline_train_losses.append(train_loss)
        baseline_val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{epochs} | Baseline (LR={best_lr_baseline}) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        # Dual-Stream
        train_loss = train_model(dual_stream_model, dl_train, dual_stream_optimizer, criterion)
        val_loss = evaluate_model(dual_stream_model, dl_test, criterion)
        dual_stream_train_losses.append(train_loss)
        dual_stream_val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{epochs} | Dual-Stream (LR={best_lr_dual_stream}) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(baseline_train_losses, label=f'Baseline Train Loss (LR={best_lr_baseline})')
    plt.plot(baseline_val_losses, label=f'Baseline Val Loss (LR={best_lr_baseline})', linestyle='--')
    plt.plot(dual_stream_train_losses, label=f'Dual-Stream Train Loss (LR={best_lr_dual_stream})')
    plt.plot(dual_stream_val_losses, label=f'Dual-Stream Val Loss (LR={best_lr_dual_stream})', linestyle='--')
    plt.title('Loss Comparison: Baseline MLP vs. Dual-Stream Network (Tuned LRs)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('dual_stream_network/loss_comparison.png')
    print("Plot saved to dual_stream_network/loss_comparison.png")

if __name__ == '__main__':
    main()
