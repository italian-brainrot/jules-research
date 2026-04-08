import torch
import matplotlib.pyplot as plt
from model import MorphologyNet
import json
import os

def visualize_kernels():
    if not os.path.exists("differentiable_morphology_experiment/results.json"):
        print("Results not found.")
        return

    # Load best LR for morphology
    with open("differentiable_morphology_experiment/results.json", "r") as f:
        results = json.load(f)

    if "morphology" not in results:
        print("Morphology results not found.")
        return

    best_lr = results["morphology"]["best_lr"]

    # Load the trained model
    model = MorphologyNet(input_dim=40, num_classes=10, num_kernels=8, kernel_size=5)
    if os.path.exists("differentiable_morphology_experiment/morphology_model.pt"):
        model.load_state_dict(torch.load("differentiable_morphology_experiment/morphology_model.pt"))
    else:
        print("Warning: Trained morphology model not found. Visualizing random kernels.")

    from mnist1d.data import make_dataset, get_dataset_args
    defaults = get_dataset_args()
    defaults.num_samples = 1000
    data = make_dataset(defaults)
    X = torch.tensor(data['x'][0:1]).float() # One sample

    model.eval()

    with torch.no_grad():
        x = X.unsqueeze(1)
        opening = model.opening(x)
        closing = model.closing(x)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(X[0].numpy(), label='Original')
    plt.title("Original Signal")
    plt.legend()

    plt.subplot(3, 1, 2)
    for i in range(4):
        plt.plot(opening[0, i].numpy(), label=f'Kernel {i}')
    plt.title("Opening Transformations")
    plt.legend()

    plt.subplot(3, 1, 3)
    for i in range(4):
        plt.plot(closing[0, i].numpy(), label=f'Kernel {i}')
    plt.title("Closing Transformations")
    plt.legend()

    plt.tight_layout()
    plt.savefig("differentiable_morphology_experiment/transformations.png")
    plt.close()

    # Also visualize kernels themselves
    kernels_op = model.opening.erosion.kernel.detach().numpy()
    num_kernels = kernels_op.shape[0]
    plt.figure(figsize=(15, 6))
    for i in range(min(8, num_kernels)):
        plt.subplot(2, 4, i+1)
        plt.plot(kernels_op[i, 0])
        plt.title(f"Kernel {i}")
    plt.suptitle("Morphological Kernels (Opening-Erosion Structural Elements)")
    plt.savefig("differentiable_morphology_experiment/kernels.png")
    plt.close()

if __name__ == "__main__":
    visualize_kernels()
