import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import numpy as np
import os

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test'])
    return X_train, y_train, X_test, y_test

def train_model(model, train_loader, test_loader, epochs=50, lr=1e-3, weight_decay=1e-4, sparsity_lambda=0.0, verbose=False):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Sparsity penalty for SoftWalshLayer
            if sparsity_lambda > 0:
                for name, param in model.named_parameters():
                    if 'swl.w' in name:
                        # Encourage w to be near 0
                        loss += sparsity_lambda * torch.mean(torch.abs(torch.tanh(param)))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        acc = correct / total
        if acc > best_acc:
            best_acc = acc

        history.append(acc)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Acc: {acc:.4f}")

    return best_acc, history

if __name__ == "__main__":
    from model import SoftWalshNetwork, MLP
    X_train, y_train, X_test, y_test = get_data()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=64, shuffle=False)

    print("Testing MLP...")
    mlp = MLP(40, 128, 10)
    mlp_acc = train_model(mlp, train_loader, test_loader, epochs=10, verbose=True)
    print(f"MLP Best Acc: {mlp_acc:.4f}")

    print("\nTesting SWN...")
    swn = SoftWalshNetwork(40, 128, 10)
    swn_acc = train_model(swn, train_loader, test_loader, epochs=10, verbose=True)
    print(f"SWN Best Acc: {swn_acc:.4f}")
