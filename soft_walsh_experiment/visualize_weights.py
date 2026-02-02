import torch
import matplotlib.pyplot as plt
from model import SoftWalshNetwork
import os

def visualize():
    nt = 512
    model = SoftWalshNetwork(40, nt, 10, deep=False)
    model.load_state_dict(torch.load("soft_walsh_experiment/swn_weights.pt"))

    w = torch.tanh(model.swl.w).detach().cpu()

    plt.figure(figsize=(10, 8))
    plt.imshow(w.numpy()[:100], aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='Weight value')
    plt.title("Learned Soft Walsh Weights (w) - first 100 terms")
    plt.xlabel("Input Feature")
    plt.ylabel("Term Index")
    plt.savefig("soft_walsh_experiment/weights_vis.png")

    plt.figure()
    plt.hist(w.numpy().flatten(), bins=50)
    plt.title("Distribution of weights w")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig("soft_walsh_experiment/weights_hist.png")

    # Check sparsity
    sparsity = (torch.abs(w) < 0.1).float().mean().item()
    print(f"Sparsity (weights < 0.1): {sparsity:.4f}")

if __name__ == "__main__":
    visualize()
