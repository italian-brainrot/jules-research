import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import mnist1d.data as mnist1d_data
from optimizer import EvolvedOptimizer
from evolve import SimpleNN
from gp import func_map

# Define the evolved optimizer expression and tuned learning rate here
# These should be the output from the evolution and tuning processes
evolved_expr_str = "sqrt(mul(m, add(m, mul(sqrt(v), neg(one)))))"
evolved_lr = 0.09964885718367121

# Define the tuned learning rate for Adam
adam_lr = 0.008109354859476139

def train(optimizer_name, lr, train_loader, test_loader, epochs=10):
    model = SimpleNN()
    loss_fn = nn.CrossEntropyLoss()

    if optimizer_name == "evolved":
        try:
            optimizer = EvolvedOptimizer(model.parameters(), evolved_expr_str, lr=lr)
        except Exception as e:
            print(f"Error creating optimizer: {e}")
            return [], []
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Unknown optimizer")

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = loss_fn(output, target)
                epoch_test_loss += loss.item()

        test_losses.append(epoch_test_loss / len(test_loader))
        print(f"Epoch {epoch+1}/{epochs}, {optimizer_name.capitalize()} Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

    return train_losses, test_losses

def main():
    # Load mnist1d data
    args = mnist1d_data.get_dataset_args()
    data = mnist1d_data.get_dataset(args)
    X_train, y_train = torch.from_numpy(data['x']).float(), torch.from_numpy(data['y']).long()
    X_test, y_test = torch.from_numpy(data['x_test']).float(), torch.from_numpy(data['y_test']).long()
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Train and evaluate Evolved Optimizer
    evolved_train_losses, evolved_test_losses = train("evolved", evolved_lr, train_loader, test_loader)

    # Train and evaluate Adam Optimizer
    adam_train_losses, adam_test_losses = train("adam", adam_lr, train_loader, test_loader)

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(evolved_train_losses, label="Evolved Optimizer Train Loss")
    plt.plot(evolved_test_losses, label="Evolved Optimizer Test Loss", linestyle='--')
    plt.plot(adam_train_losses, label="Adam Optimizer Train Loss")
    plt.plot(adam_test_losses, label="Adam Optimizer Test Loss", linestyle='--')
    plt.title("Optimizer Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("optimizer_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
