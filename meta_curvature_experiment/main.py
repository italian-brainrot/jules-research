import torch
import torch.nn as nn
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import matplotlib.pyplot as plt
import copy
import os

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
        for inputs, targets in dl_train:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            def closure():
                return loss

            optimizer.zero_grad()
            if isinstance(optimizer, MetaCurvatureLR):
                loss.backward(create_graph=True)
                optimizer.step(closure)
            else:
                loss.backward()
                optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in dl_test:
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
        test_losses.append(test_loss / len(dl_test))
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

    return train_losses, test_losses

def main():
    # --- Data Loading ---
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])

    dl_train = TensorDataLoader((X_train, y_train), batch_size=32, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=32, shuffle=False)

    # --- Model Initialization ---
    base_model = get_model()

    # --- Baseline: Adam ---
    print("Training with Adam...")
    model_adam = copy.deepcopy(base_model)
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-3)
    train_losses_adam, test_losses_adam = train(model_adam, optimizer_adam, dl_train, dl_test)

    # --- Experiment: MetaCurvatureLR(Adam) ---
    print("\nTraining with MetaCurvatureLR(Adam)...")
    model_meta = copy.deepcopy(base_model)
    base_optimizer_meta = torch.optim.Adam(model_meta.parameters(), lr=1e-3)
    optimizer_meta = MetaCurvatureLR(base_optimizer_meta, update_freq=10, alpha=1.0)
    train_losses_meta, test_losses_meta = train(model_meta, optimizer_meta, dl_train, dl_test)

    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_adam, label='Adam')
    plt.plot(train_losses_meta, label='MetaCurvatureLR(Adam)')
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(test_losses_adam, label='Adam')
    plt.plot(test_losses_meta, label='MetaCurvatureLR(Adam)')
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot in the experiment's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'loss_comparison.png'))
    plt.show()

if __name__ == "__main__":
    main()
