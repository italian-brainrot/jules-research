import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import spearmanr
from mnist1d.data import get_dataset_args, make_dataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_data(num_samples=300):
    print(f"Loading {num_samples} samples from mnist1d...")
    args = get_dataset_args()
    args.num_samples = num_samples
    data = make_dataset(args)
    X = torch.tensor(data['x'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.long)
    return X, y

def compute_hard_ranks(D):
    # D: (N, N)
    # Returns ranks of each row. Smallest distance gets rank 0.
    return D.argsort(dim=1).argsort(dim=1).float()

class SoftRankMDS(nn.Module):
    def __init__(self, N, initial_alpha=10.0, Z_init=None):
        super().__init__()
        if Z_init is not None:
            self.Z = nn.Parameter(torch.tensor(Z_init, dtype=torch.float32))
        else:
            self.Z = nn.Parameter(torch.randn(N, 2) * 0.01)
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))

    def forward(self):
        N = self.Z.shape[0]
        d = torch.cdist(self.Z, self.Z)
        # Memory efficient soft rank computation for N=500
        # diff: (N, N, N) -> diff[i, j, k] = d[i, j] - d[i, k]
        diff = d.unsqueeze(2) - d.unsqueeze(1)
        soft_ranks = torch.sigmoid(self.alpha * diff).sum(dim=2)
        return soft_ranks

def train_soft_rank_mds(X, target_ranks, num_steps=500, lr=0.01):
    N = X.shape[0]
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(X.numpy())

    model = SoftRankMDS(N, Z_init=Z_pca)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in range(num_steps):
        optimizer.zero_grad()
        soft_ranks = model()
        loss = nn.MSELoss()(soft_ranks, target_ranks)
        loss.backward()
        optimizer.step()
        if (i+1) % 200 == 0:
            print(f"  SR-MDS Step {i+1}, Loss: {loss.item():.4f}, Alpha: {model.alpha.item():.4f}")

    return model.Z.detach().numpy()

def train_metric_mds(X, D_high, num_steps=500, lr=0.01):
    N = X.shape[0]
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(X.numpy())

    Z = nn.Parameter(torch.tensor(Z_pca, dtype=torch.float32))
    optimizer = optim.Adam([Z], lr=lr)

    # Normalize D_high to have similar scale to PCA Z
    D_target = D_high / D_high.std() * torch.tensor(Z_pca).std()

    for i in range(num_steps):
        optimizer.zero_grad()
        d_low = torch.cdist(Z, Z)
        loss = nn.MSELoss()(d_low, D_target)
        loss.backward()
        optimizer.step()
        if (i+1) % 200 == 0:
            print(f"  Metric MDS Step {i+1}, Loss: {loss.item():.4f}")

    return Z.detach().numpy()

def evaluate(Z, y, D_high_np):
    # KNN accuracy
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Z, y)
    acc = knn.score(Z, y)

    # Spearman correlation
    N = Z.shape[0]
    D_low = np.linalg.norm(Z[:, None] - Z[None, :], axis=2)
    flat_high = D_high_np.flatten()
    flat_low = D_low.flatten()
    # We use a subset of pairs for Spearman to be fast if needed,
    # but for N=500, all pairs is fine (250,000)
    corr, _ = spearmanr(flat_high, flat_low)

    return acc, corr

def plot_embedding(Z, y, title, filename):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def main():
    os.makedirs("soft_rank_mds_experiment", exist_ok=True)
    X, y = get_data(num_samples=300)
    D_high = torch.cdist(X, X)
    D_high_np = D_high.numpy()
    target_ranks = compute_hard_ranks(D_high)

    results = {}

    # PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(X.numpy())
    results['PCA'] = Z_pca

    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    Z_tsne = tsne.fit_transform(X.numpy())
    results['t-SNE'] = Z_tsne

    # Metric MDS
    print("Running Metric MDS...")
    Z_mmds = train_metric_mds(X, D_high)
    results['Metric MDS'] = Z_mmds

    # Soft Rank MDS
    print("Running Soft Rank MDS...")
    Z_srmds = train_soft_rank_mds(X, target_ranks)
    results['Soft Rank MDS'] = Z_srmds

    print("\nEvaluation:")
    print(f"{'Method':<20} | {'KNN Acc':<10} | {'Spearman':<10}")
    print("-" * 45)

    with open("soft_rank_mds_experiment/results.txt", "w") as f:
        f.write(f"{'Method':<20} | {'KNN Acc':<10} | {'Spearman':<10}\n")
        f.write("-" * 45 + "\n")
        for name, Z in results.items():
            acc, corr = evaluate(Z, y, D_high_np)
            print(f"{name:<20} | {acc:<10.4f} | {corr:<10.4f}")
            f.write(f"{name:<20} | {acc:<10.4f} | {corr:<10.4f}\n")
            plot_embedding(Z, y, f"{name} on MNIST-1D", f"soft_rank_mds_experiment/{name.lower().replace(' ', '_')}.png")

if __name__ == "__main__":
    main()
