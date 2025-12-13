
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import matplotlib.pyplot as plt
import pickle
import os
from gp_core import Node, evaluate_expression, FUNCTION_NAMES

# --- Load the evolved expression ---
best_evolved_expression = None
expression_path = "gplrs_experiment/best_expression.pkl"
if os.path.exists(expression_path):
    with open(expression_path, "rb") as f:
        best_evolved_expression = pickle.load(f)
    print(f"Loaded evolved expression: {best_evolved_expression}")
else:
    print("No evolved expression found. Running without GP scheduler.")

# --- CONFIGURATIONS ---
NUM_EPOCHS = 30
LEARNING_RATE_CAP = 1.0

# --- DATASET ---
def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 4000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)
    return dl_train, dl_test

# --- NEURAL NETWORK ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.layers(x)

# --- TRAINING LOOP ---
def train_model(scheduler_name, train_loader, val_loader, scheduler_fn=None, evolved_expr=None):
    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    if scheduler_fn:
        scheduler = scheduler_fn(optimizer)

    history = {'train_loss': [], 'val_acc': [], 'lr': []}

    print(f"--- Training with {scheduler_name} ---")

    last_batch_loss = 0.01 # Initial loss for LR calculation

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_epoch_loss = 0
        for inputs, targets in train_loader:
            if scheduler_name == "Evolved GP":
                # Calculate LR using the evolved expression from the *previous* batch
                lr = evaluate_expression(evolved_expr, epoch=epoch, loss=last_batch_loss, val_loss=0)
                lr = max(0.0, min(lr, LEARNING_RATE_CAP))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            last_batch_loss = loss.item()
            total_epoch_loss += last_batch_loss

        history['train_loss'].append(total_epoch_loss / len(train_loader))
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if scheduler_name != "Evolved GP" and scheduler:
            scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_acc = correct / total
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {history['train_loss'][-1]:.4f}, Val Acc: {val_acc:.4f}, LR: {history['lr'][-1]:.6f}")

    return history

def main():
    train_loader, val_loader = get_data()

    # --- DEFINE SCHEDULERS ---
    schedulers = {
        "Constant LR (0.001)": lambda opt: None,
        "Step Decay": lambda opt: StepLR(opt, step_size=10, gamma=0.1),
        "Cosine Annealing": lambda opt: CosineAnnealingLR(opt, T_max=NUM_EPOCHS),
    }

    # Add the Evolved GP scheduler only if the expression was loaded
    if best_evolved_expression:
        schedulers["Evolved GP"] = lambda opt: None # Handled inside the training loop

    results = {}

    # Run training for each scheduler
    for name, fn in schedulers.items():
        if name == "Evolved GP":
            results[name] = train_model(name, train_loader, val_loader, evolved_expr=best_evolved_expression)
        else:
            results[name] = train_model(name, train_loader, val_loader, scheduler_fn=fn)

    # --- PLOT RESULTS ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
    fig.suptitle('Learning Rate Scheduler Comparison')

    # Plot Validation Accuracy
    ax1.set_title('Validation Accuracy vs. Epochs')
    for name, data in results.items():
        ax1.plot(data['val_acc'], label=name)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Plot Training Loss
    ax2.set_title('Training Loss vs. Epochs')
    for name, data in results.items():
        ax2.plot(data['train_loss'], label=name)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.legend()
    ax2.grid(True)

    # Plot Learning Rate
    ax3.set_title('Learning Rate vs. Epochs')
    for name, data in results.items():
        ax3.plot(data['lr'], label=name)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('gplrs_experiment/comparison.png')
    print("\nComparison plot saved to gplrs_experiment/comparison.png")

if __name__ == "__main__":
    main()
