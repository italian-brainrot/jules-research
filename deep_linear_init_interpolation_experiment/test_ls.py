import torch
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)

    X_train = torch.tensor(data['x']).float()
    y_train = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()

    return X_train, y_train, X_test, y_test

def compute_ls_solution(X, y):
    # One-hot encode y
    num_classes = 10
    Y = torch.zeros(X.shape[0], num_classes)
    Y.scatter_(1, y.unsqueeze(1), 1.0)

    # Solve XW = Y  => W = (X^T X)^-1 X^T Y
    # Add small regularization for stability
    XTX = X.t() @ X
    XTX += 1e-4 * torch.eye(XTX.shape[0])
    W = torch.linalg.solve(XTX, X.t() @ Y)

    return W

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    W_ls = compute_ls_solution(X_train, y_train)
    print(f"W_ls shape: {W_ls.shape}")

    # Check accuracy of LS solution
    y_pred = X_test @ W_ls
    acc = (y_pred.argmax(dim=1) == y_test).float().mean()
    print(f"LS solution test accuracy: {acc.item():.4f}")

    # Check linear train accuracy
    y_pred_train = X_train @ W_ls
    acc_train = (y_pred_train.argmax(dim=1) == y_train).float().mean()
    print(f"LS solution train accuracy: {acc_train.item():.4f}")
