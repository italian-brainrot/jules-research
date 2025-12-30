
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import collections
from mnist1d.data import make_dataset, get_dataset_args

# --- 1. Define the Model (must match the training script) ---
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# --- 2. Evaluation Function ---
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_test).sum().item()
        return correct / len(y_test)

# --- 3. Interpolation Function ---
def interpolate_weights(state_dict1, state_dict2, alpha):
    interpolated_state_dict = collections.OrderedDict()
    for key in state_dict1:
        interpolated_state_dict[key] = (1 - alpha) * state_dict1[key] + alpha * state_dict2[key]
    return interpolated_state_dict

def main():
    # --- 4. Load Data and Snapshots ---
    defaults = get_dataset_args()
    data = make_dataset(defaults)
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.int64)

    snapshots = torch.load('weight_interpolation_experiment/snapshots.pt')
    snapshot_epochs = list(snapshots.keys())

    # --- 5. Setup Model ---
    input_size = X_test.shape[1]
    hidden_size = 256
    output_size = y_test.max().item() + 1
    model = MLP(input_size, hidden_size, output_size)

    # --- 6. Baseline Accuracy ---
    baseline_epoch = snapshot_epochs[-1]
    model.load_state_dict(snapshots[baseline_epoch])
    baseline_accuracy = evaluate(model, X_test, y_test)
    print(f"Baseline Accuracy (Epoch {baseline_epoch}): {baseline_accuracy:.4f}")

    # --- 7. Interpolation and Evaluation ---
    alphas = np.linspace(-0.5, 1.5, 21) # Check outside the [0,1] interval as well
    results = np.zeros((len(snapshot_epochs), len(snapshot_epochs), len(alphas)))

    for i, epoch1 in enumerate(snapshot_epochs):
        for j, epoch2 in enumerate(snapshot_epochs):
            if i >= j: continue # Avoid redundant calculations and self-interpolation at alpha=0.5
            for k, alpha in enumerate(alphas):
                interpolated_state = interpolate_weights(snapshots[epoch1], snapshots[epoch2], alpha)
                model.load_state_dict(interpolated_state)
                accuracy = evaluate(model, X_test, y_test)
                results[i, j, k] = accuracy
                print(f"Epochs ({epoch1}, {epoch2}), Alpha: {alpha:.2f}, Accuracy: {accuracy:.4f}")

    # --- 8. Find and Report Best Result ---
    best_accuracy = np.max(results)
    best_indices = np.unravel_index(np.argmax(results), results.shape)
    best_epoch1 = snapshot_epochs[best_indices[0]]
    best_epoch2 = snapshot_epochs[best_indices[1]]
    best_alpha = alphas[best_indices[2]]

    print("\\n--- Results ---")
    print(f"Best Interpolated Accuracy: {best_accuracy:.4f}")
    print(f"Achieved with Epochs: ({best_epoch1}, {best_epoch2}) and Alpha: {best_alpha:.2f}")
    print(f"Baseline Accuracy (Epoch {baseline_epoch}): {baseline_accuracy:.4f}")
    print(f"Improvement over baseline: {best_accuracy - baseline_accuracy:.4f}")


    # --- 9. Visualization ---
    # We will visualize the accuracy landscape for a specific pair of epochs
    best_pair_results = results[best_indices[0], best_indices[1], :]

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, best_pair_results, marker='o', linestyle='-')
    plt.title(f'Accuracy vs. Alpha for Epochs {best_epoch1} and {best_epoch2}')
    plt.xlabel('Alpha')
    plt.ylabel('Test Accuracy')
    plt.grid(True)

    # Add lines for the accuracy of the original two models
    model.load_state_dict(snapshots[best_epoch1])
    acc1 = evaluate(model, X_test, y_test)
    model.load_state_dict(snapshots[best_epoch2])
    acc2 = evaluate(model, X_test, y_test)
    plt.axvline(x=0, color='r', linestyle='--', label=f'Epoch {best_epoch1} Acc: {acc1:.4f}')
    plt.axvline(x=1, color='g', linestyle='--', label=f'Epoch {best_epoch2} Acc: {acc2:.4f}')

    plt.legend()
    plt.savefig('weight_interpolation_experiment/interpolation_accuracy.png')
    print("\\nPlot saved to weight_interpolation_experiment/interpolation_accuracy.png")


if __name__ == "__main__":
    main()
