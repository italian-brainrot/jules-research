import torch
import torch.nn as nn
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import matplotlib.pyplot as plt
import copy
import os

from optimizer import GPE

# --- 1. Dataset Setup ---
defaults = get_dataset_args()
defaults.num_samples = 10000
data = make_dataset(defaults)

X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)

dl_train = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)
dl_test = TensorDataLoader((X_test, y_test), batch_size=1024)

# --- 2. Model Definition ---
def create_model():
    return nn.Sequential(
        nn.Linear(X_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

# --- 3. Training and Evaluation Loop ---
def train_and_evaluate(optimizer, model, epochs=50):
    criterion = nn.CrossEntropyLoss()
    val_losses = []

    for epoch in range(epochs):
        model.train()
        for inputs, targets in dl_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in dl_test:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

        val_loss = total_loss / len(dl_test.data[0])
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")

    return val_losses

# --- 4. Optimizer Comparison ---
if __name__ == "__main__":
    # Ensure fair comparison by starting with the same initial weights
    initial_model = create_model()

    # --- Adam Optimizer ---
    print("--- Training with Adam ---")
    model_adam = copy.deepcopy(initial_model)
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-3)
    losses_adam = train_and_evaluate(optimizer_adam, model_adam)

    # --- GPE(Adam) Optimizer ---
    print("\n--- Training with GPE(Adam) ---")
    model_gpe = copy.deepcopy(initial_model)
    base_optimizer_gpe = torch.optim.Adam(model_gpe.parameters(), lr=1e-3)
    optimizer_gpe = GPE(model_gpe.parameters(), base_optimizer_gpe, history_size=10, degree=2, alpha=0.4)
    losses_gpe = train_and_evaluate(optimizer_gpe, model_gpe)

    # --- 5. Plotting Results ---
    plt.figure(figsize=(10, 6))
    plt.plot(losses_adam, label='Adam')
    plt.plot(losses_gpe, label='GPE(Adam)')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Adam vs. GPE(Adam) Optimizer Performance')
    plt.legend()
    plt.grid(True)

    # Save the plot in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'comparison_plot.png'))

    print("\nComparison plot saved to 'comparison_plot.png'")
