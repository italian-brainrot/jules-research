import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import matplotlib.pyplot as plt
import copy

from .model import MLP
from .optimizer import GNSAdam

def run_experiment(optimizer_class, model, train_loader, test_loader, epochs, **optimizer_kwargs):
    """Runs a training and evaluation experiment for a given optimizer."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    test_accuracies = []

    for epoch in range(epochs):
        # Training loop
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()

        # Evaluation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs.float())
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f"Epoch {epoch+1}/{epochs}, Optimizer: {optimizer_class.__name__}, Test Accuracy: {accuracy:.2f}%")

    return test_accuracies

def main():
    # --- Hyperparameters ---
    EPOCHS = 15
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128

    # --- Dataset ---
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X_train, y_train = torch.tensor(data['x']), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]), torch.tensor(data['y_test'])

    dl_train = TensorDataLoader((X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # --- Model Initialization ---
    # Ensure both models start with the exact same weights for a fair comparison
    model_adam = MLP()
    model_gns = MLP()
    model_gns.load_state_dict(copy.deepcopy(model_adam.state_dict()))

    # --- Run Experiments ---
    print("--- Starting Adam Experiment ---")
    adam_history = run_experiment(
        optim.Adam, model_adam, dl_train, dl_test, EPOCHS, lr=LEARNING_RATE
    )

    print("\n--- Starting GNS-Adam Experiment ---")
    gns_adam_history = run_experiment(
        GNSAdam, model_gns, dl_train, dl_test, EPOCHS, base_optimizer=optim.Adam, lr=LEARNING_RATE
    )

    # --- Plot Results ---
    plt.figure(figsize=(10, 6))
    plt.plot(adam_history, label="Adam", marker='o')
    plt.plot(gns_adam_history, label="GNS-Adam", marker='x')
    plt.title("Optimizer Comparison: Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("gns_adam_experiment/comparison_plot.png")
    print("\nComparison plot saved to gns_adam_experiment/comparison_plot.png")

if __name__ == "__main__":
    main()
