import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from spectral_class_divergence_experiment.train import MLP, compute_input_gradients

def visualize_signatures():
    defaults = get_dataset_args()
    defaults.num_samples = 2000
    data = make_dataset(defaults)
    X = torch.tensor(data['x']).float()
    y = torch.tensor(data['y']).long()

    model = MLP()
    # Random model just to see what they look like, or I could train it for a few epochs
    # Let's train for 5 epochs to get some signal
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        outputs = model(X)
        loss = F.cross_entropy(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    X.requires_grad_(True)
    input_grads = compute_input_gradients(model, X, y)

    freqs = torch.fft.rfft(input_grads, dim=1)
    power = torch.abs(freqs)**2
    power_norm = power / (power.sum(dim=1, keepdim=True) + 1e-8)

    plt.figure(figsize=(10, 6))
    for c in range(10):
        mask = (y == c)
        if mask.any():
            class_mean = power_norm[mask].mean(dim=0).detach().numpy()
            plt.plot(class_mean, label=f"Class {c}")

    plt.title("Class-wise Spectral Signatures of Input Gradients")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Normalized Power")
    plt.legend()
    plt.savefig("spectral_class_divergence_experiment/spectral_signatures.png")

if __name__ == "__main__":
    visualize_signatures()
