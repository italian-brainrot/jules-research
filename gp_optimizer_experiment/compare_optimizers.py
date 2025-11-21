import numpy as np
import torch
import pickle
from model import LogisticRegression
from gp_framework import GeneticProgrammingOptimizer

# Load the dataset
data = np.load('logistic_regression_dataset.npz')
X = data['X']
y = data['y']

# Load the best evolved optimizer
with open('best_optimizer.pkl', 'rb') as f:
    best_optimizer_tree = pickle.load(f)

# Convert data to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define the learning rate range to search
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0]

def find_best_lr(model_class, optimizer_class, X_tensor, y_tensor, optimizer_args={}):
    best_lr = None
    best_loss = float('inf')

    for lr in learning_rates:
        model = model_class(X_tensor.shape[1])
        optimizer = optimizer_class(model.parameters(), lr=lr, **optimizer_args)
        criterion = torch.nn.BCELoss()

        for epoch in range(500):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            if torch.isnan(loss) or torch.isinf(loss):
                loss = torch.tensor(float('inf'))
                break
            loss.backward()
            optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_lr = lr
        print(f"  LR: {lr}, Final Loss: {loss.item():.4f}")


    return best_lr, best_loss

# Find the best learning rate for the evolved optimizer
print("Searching for the best learning rate for the evolved optimizer...")
best_lr_gp, best_loss_gp = find_best_lr(
    LogisticRegression,
    GeneticProgrammingOptimizer,
    X_tensor,
    y_tensor,
    optimizer_args={'gp_tree': best_optimizer_tree}
)

print(f"\nBest LR for Evolved Optimizer: {best_lr_gp}, Loss: {best_loss_gp:.4f}")


# Find the best learning rate for Adam
print("\nSearching for the best learning rate for Adam optimizer...")
best_lr_adam, best_loss_adam = find_best_lr(
    LogisticRegression,
    torch.optim.Adam,
    X_tensor,
    y_tensor
)

print(f"\nBest LR for Adam: {best_lr_adam}, Loss: {best_loss_adam:.4f}")

print("\n--- Comparison ---")
print(f"Evolved Optimizer: Loss = {best_loss_gp:.4f} (at LR={best_lr_gp})")
print(f"Adam Optimizer:    Loss = {best_loss_adam:.4f} (at LR={best_lr_adam})")
