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

    # Add bias by appending 1s to X
    X_aug = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)

    # Solve X_aug W_full = Y  => W_full = (X_aug^T X_aug)^-1 X_aug^T Y
    XTX = X_aug.t() @ X_aug
    XTX += 1e-4 * torch.eye(XTX.shape[0])
    W_full = torch.linalg.solve(XTX, X_aug.t() @ Y)

    W = W_full[:-1, :]
    b = W_full[-1, :]

    return W, b

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()
    W_ls, b_ls = compute_ls_solution(X_train, y_train)

    # Check accuracy
    y_pred = X_test @ W_ls + b_ls
    acc = (y_pred.argmax(dim=1) == y_test).float().mean()
    print(f"LS (with bias) test accuracy: {acc.item():.4f}")

    y_pred_train = X_train @ W_ls + b_ls
    acc_train = (y_pred_train.argmax(dim=1) == y_train).float().mean()
    print(f"LS (with bias) train accuracy: {acc_train.item():.4f}")
