import torch
from model import DSBMLP, FSBMLP
from train import get_data
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_sparsity(model, X, model_type="DSB"):
    model.eval()
    with torch.no_grad():
        mean_sparsity_per_layer = []
        std_sparsity_per_layer = []

        # model.layers contains [Linear, DSB/FSB, Linear, DSB/FSB, ...]
        layer_activations = []
        x = X
        for i in range(0, len(model.layers), 2):
            lin = model.layers[i]
            bottleneck = model.layers[i+1]
            x = lin(x)
            x = bottleneck(x)

            # x has shape (N, hidden_dim)
            sparsity_per_sample = (x == 0).float().mean(dim=1) # (N,)

            mean_sparsity_per_layer.append(sparsity_per_sample.mean().item())
            std_sparsity_per_layer.append(sparsity_per_sample.std().item())

    return mean_sparsity_per_layer, std_sparsity_per_layer

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    dsb_means = []
    dsb_stds = []
    fsb_means = []
    fsb_stds = []

    # Try to load models if they exist
    for seed in range(3):
        dsb_path = f"dynamic_sparse_bottleneck_experiment/DSB_seed{seed}.pt"
        fsb_path = f"dynamic_sparse_bottleneck_experiment/FSB_seed{seed}.pt"

        if os.path.exists(dsb_path):
            model = DSBMLP()
            model.load_state_dict(torch.load(dsb_path))
            m, s = analyze_sparsity(model, X_test, "DSB")
            dsb_means.append(m)
            dsb_stds.append(s)

        if os.path.exists(fsb_path):
            model = FSBMLP()
            model.load_state_dict(torch.load(fsb_path))
            m, s = analyze_sparsity(model, X_test, "FSB")
            fsb_means.append(m)
            fsb_stds.append(s)

    if dsb_means and fsb_means:
        dsb_avg_mean = np.mean(dsb_means, axis=0)
        dsb_avg_std = np.mean(dsb_stds, axis=0)
        fsb_avg_mean = np.mean(fsb_means, axis=0)
        fsb_avg_std = np.mean(fsb_stds, axis=0)

        layers = range(1, len(dsb_avg_mean) + 1)

        # Plot Mean Sparsity
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(layers, dsb_avg_mean * 100, marker='o', label='DSB (Dynamic)')
        plt.plot(layers, fsb_avg_mean * 100, marker='s', label='FSB (Fixed)')
        plt.xlabel('Layer')
        plt.ylabel('Average Sparsity (%)')
        plt.title('Average Layer-wise Sparsity')
        plt.xticks(layers)
        plt.legend()
        plt.grid(True)

        # Plot Sparsity SD across inputs
        plt.subplot(1, 2, 2)
        plt.plot(layers, dsb_avg_std * 100, marker='o', color='blue', label='DSB')
        plt.plot(layers, fsb_avg_std * 100, marker='s', color='orange', label='FSB')
        plt.xlabel('Layer')
        plt.ylabel('Sparsity SD (%)')
        plt.title('Sparsity Variation across Inputs')
        plt.xticks(layers)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('dynamic_sparse_bottleneck_experiment/sparsity.png')
        plt.close()
        print("Sparsity analysis completed and plot saved.")

        print("\nSparsity Summary:")
        for i in range(len(layers)):
            print(f"Layer {i+1}:")
            print(f"  DSB Mean Sparsity: {dsb_avg_mean[i]*100:.2f}%, SD across inputs: {dsb_avg_std[i]*100:.2f}%")
            print(f"  FSB Mean Sparsity: {fsb_avg_mean[i]*100:.2f}%, SD across inputs: {fsb_avg_std[i]*100:.2f}%")
    else:
        print("Models not found for sparsity analysis. Run compare.py first.")
