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

# Train with the evolved optimizer
print("Training with evolved optimizer...")
model_gp = LogisticRegression(X.shape[1])
optimizer_gp = GeneticProgrammingOptimizer(model_gp.parameters(), best_optimizer_tree)
criterion = torch.nn.BCELoss()

for epoch in range(500):
    optimizer_gp.zero_grad()
    outputs = model_gp(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer_gp.step()

print(f"Final loss with evolved optimizer: {loss.item()}")

# Train with Adam
print("\nTraining with Adam optimizer...")
model_adam = LogisticRegression(X.shape[1])
optimizer_adam = torch.optim.Adam(model_adam.parameters())
criterion = torch.nn.BCELoss()

for epoch in range(500):
    optimizer_adam.zero_grad()
    outputs = model_adam(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer_adam.step()

print(f"Final loss with Adam optimizer: {loss.item()}")
