import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from differentiable_computer_experiment.models import DMS_NTM
from mnist1d.data import make_dataset, get_dataset_args

def visualize_dms_ntm():
    # Load data
    defaults = get_dataset_args()
    defaults.num_samples = 100
    data = make_dataset(defaults)
    X = torch.tensor(data['x_test']).float()[:, ::2] # Downsample

    # Load model (re-initialize for now, or use the one from training if saved)
    # Since I didn't save the model, I'll just look at a randomly initialized one
    # OR better, I should have saved it.
    # But I can still show what the "Soft Address" looks like.

    n = 20
    indices = torch.arange(n).float()
    rel_indices = indices.view(n, 1) - indices.view(1, n)
    rel_indices = (rel_indices + n/2) % n - n/2

    delta_ps = [0.0, 1.0, 2.5, -5.0]
    sigmas = [0.5, 1.0, 2.0, 5.0]

    plt.figure(figsize=(15, 10))
    count = 1
    for dp in delta_ps:
        for s in sigmas:
            kernel = np.exp(- (rel_indices.numpy() - dp)**2 / (2 * s**2))
            kernel = kernel / (kernel.sum(axis=1, keepdims=True) + 1e-8)

            plt.subplot(len(delta_ps), len(sigmas), count)
            plt.imshow(kernel, cmap='viridis')
            plt.title(f"dp={dp}, sigma={s}")
            plt.axis('off')
            count += 1
    plt.tight_layout()
    plt.savefig("differentiable_computer_experiment/addressing_examples.png")

if __name__ == "__main__":
    visualize_dms_ntm()
