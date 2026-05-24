import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
from mnist1d.data import make_dataset, get_dataset_args
from music import MUSICMLP, BaselineMLP

def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test'])
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    with open("differentiable_music_experiment/results.json", "r") as f:
        data = json.load(f)

    results = data["results"]
    best_params = data["best_params"]

    # Bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), [v * 100 for v in results.values()])
    plt.ylabel("Accuracy (%)")
    plt.title("Model Comparison on MNIST-1D")
    plt.savefig("differentiable_music_experiment/comparison.png")

    # Pseudospectra
    params = best_params["music"]
    music_layer = MUSICMLP(40, params['window_size'], params['hidden_dim'], 10, num_freqs=params['num_freqs']).music

    plt.figure(figsize=(12, 10))
    for i in range(5):
        plt.subplot(5, 2, 2*i + 1)
        plt.plot(X_test[i].numpy())
        plt.title(f"Signal (Class {y_test[i].item()})")

        plt.subplot(5, 2, 2*i + 2)
        with torch.no_grad():
            ps = music_layer(X_test[i].unsqueeze(0)).squeeze(0).numpy()
        plt.plot(ps)
        plt.title("MUSIC Pseudospectrum (Log Scale)")

    plt.tight_layout()
    plt.savefig("differentiable_music_experiment/pseudospectra.png")
    print("Plots regenerated.")
