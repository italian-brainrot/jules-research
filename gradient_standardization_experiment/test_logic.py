import torch
from gradient_standardization_experiment.train import centralize_gradient, standardize_gradient

def test_centralize_gradient():
    # Create a dummy gradient for a Linear layer (out_features=3, in_features=4)
    grad = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
        [-1.0, -2.0, -3.0, -4.0]
    ])

    centralized = centralize_gradient(grad)

    # Check that each row now has mean 0
    row_means = centralized.mean(dim=1)
    assert torch.allclose(row_means, torch.zeros_like(row_means), atol=1e-6)

    # Check that the relative differences between elements in a row are preserved
    # (grad - mean) - (grad' - mean) = grad - grad'
    diff_original = grad[:, 1] - grad[:, 0]
    diff_centralized = centralized[:, 1] - centralized[:, 0]
    assert torch.allclose(diff_original, diff_centralized)

def test_standardize_gradient():
    # Create a dummy gradient
    grad = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
        [-1.0, -2.0, -3.0, -4.0]
    ])

    standardized = standardize_gradient(grad)

    # Check mean 0
    row_means = standardized.mean(dim=1)
    assert torch.allclose(row_means, torch.zeros_like(row_means), atol=1e-6)

    # Check std 1 (unbiased std)
    row_stds = standardized.std(dim=1)
    assert torch.allclose(row_stds, torch.ones_like(row_stds), atol=1e-6)

if __name__ == "__main__":
    test_centralize_gradient()
    test_standardize_gradient()
    print("All tests passed!")
