import numpy as np
import torch
import subprocess
from model import LogisticRegression
from curve_fit_optimizer import CurveFitOptimizer

# Generate the dataset
try:
    subprocess.run(['python', 'gradient_compression_optimizer/generate_dataset.py'], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error generating dataset: {e}")
    exit(1)


# Load the dataset
data = np.load('gradient_compression_optimizer/logistic_regression_dataset.npz')
X = data['X']
y = data['y']

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

        # Special handling for CurveFitOptimizer to pass base_optimizer correctly
        if optimizer_class == CurveFitOptimizer:
            base_optimizer = optimizer_args.get('base_optimizer')
            if base_optimizer is None:
                raise ValueError("CurveFitOptimizer requires a 'base_optimizer' in optimizer_args")

            optimizer = optimizer_class(model.parameters(), base_optimizer=base_optimizer, lr=lr)
        else:
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

# Find the best learning rate for the CurveFit optimizer
print("Searching for the best learning rate for the CurveFit optimizer...")
best_lr_curvefit, best_loss_curvefit = find_best_lr(
    LogisticRegression,
    CurveFitOptimizer,
    X_tensor,
    y_tensor,
    optimizer_args={'base_optimizer': torch.optim.Adam}
)

print(f"\nBest LR for CurveFit Optimizer: {best_lr_curvefit}, Loss: {best_loss_curvefit:.4f}")


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
print(f"CurveFit Optimizer: Loss = {best_loss_curvefit:.4f} (at LR={best_lr_curvefit})")
print(f"Adam Optimizer:     Loss = {best_loss_adam:.4f} (at LR={best_lr_adam})")
