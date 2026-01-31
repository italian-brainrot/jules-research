import torch
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args

def check():
    args = get_dataset_args()
    args.num_samples = 10000
    data = make_dataset(args)

    X = torch.tensor(data['x'], dtype=torch.float32)
    mean = X.mean(dim=0)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)

    eigvals, _ = torch.linalg.eigh(cov)
    print("Eigenvalues (sorted):")
    print(eigvals)

    print("\nMin eigval:", eigvals.min().item())
    print("Max eigval:", eigvals.max().item())
    print("Condition number:", (eigvals.max() / eigvals.min()).item())

if __name__ == "__main__":
    check()
