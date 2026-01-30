import torch
import torch.nn.functional as F
import numpy as np

def test_apls_logic():
    num_classes = 3
    feature_dim = 4
    epsilon = 0.1
    temp = 1.0

    # Mock centroids
    # Class 0 and 1 are close, Class 2 is far
    centroids = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.1, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0]
    ])

    # Mock labels
    y = torch.tensor([0, 1, 2])

    # Compute distances
    dist = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
    print(f"Distances:\n{dist}")

    dist_for_softmax = dist.clone()
    dist_for_softmax.fill_diagonal_(float('inf'))

    weights = F.softmax(-dist_for_softmax / temp, dim=1)
    print(f"Weights:\n{weights}")

    targets_matrix = weights * epsilon
    targets_matrix.scatter_(1, torch.arange(num_classes).view(-1, 1), 1 - epsilon)
    print(f"Targets Matrix:\n{targets_matrix}")

    # Check properties
    # 1. Sum of each row should be 1
    assert torch.allclose(targets_matrix.sum(dim=1), torch.tensor(1.0))

    # 2. Diagonal should be 1 - epsilon
    assert torch.allclose(torch.diag(targets_matrix), torch.tensor(1 - epsilon))

    # 3. For class 0, weight for class 1 should be higher than for class 2
    assert targets_matrix[0, 1] > targets_matrix[0, 2]

    # 4. For class 2, weights for class 0 and 1 should be similar (and small)
    assert torch.allclose(targets_matrix[2, 0], targets_matrix[2, 1], atol=1e-2)

    print("APLS logic test passed!")

if __name__ == "__main__":
    test_apls_logic()
