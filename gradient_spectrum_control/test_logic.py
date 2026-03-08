import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad

def compute_loss(params, model, x, y):
    logits = functional_call(model, params, (x.unsqueeze(0),))
    return nn.functional.cross_entropy(logits, y.unsqueeze(0))

def get_spectrum_stats(params, model, x_batch, y_batch):
    batch_size = x_batch.shape[0]

    def single_loss(p, x, y):
        return compute_loss(p, model, x, y)

    # Compute per-sample gradients
    per_sample_grads_dict = vmap(grad(single_loss), in_dims=(None, 0, 0))(params, x_batch, y_batch)

    # Flatten and concatenate
    flat_grads = []
    for p in per_sample_grads_dict.values():
        flat_grads.append(p.reshape(batch_size, -1))
    G = torch.cat(flat_grads, dim=1)  # (B, P)

    # Normalize each gradient to unit norm to focus on direction spectrum
    G_norm = torch.norm(G, dim=1, keepdim=True) + 1e-8
    G_normalized = G / G_norm

    # Gram matrix K = G_norm * G_norm^T (B, B)
    K = torch.mm(G_normalized, G_normalized.t())

    # Eigenvalues of K
    # L = torch.linalg.eigvalsh(K) # sorted ascending
    # For differentiability and stability, we can use SVD or eigvalsh
    L = torch.linalg.eigvalsh(K)
    L = torch.relu(L) # Ensure non-negative due to numerical noise

    max_eig = L[-1]
    total_eig = L.sum()
    ratio = max_eig / (total_eig + 1e-8)

    return ratio

class SimpleMLP(nn.Module):
    def __init__(self, input_size=4, hidden_size=8, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def test_logic():
    model = SimpleMLP()
    params = dict(model.named_parameters())

    batch_size = 4
    x = torch.randn(batch_size, 4)
    y = torch.randint(0, 2, (batch_size,))

    # Test 1: Random gradients
    ratio = get_spectrum_stats(params, model, x, y)
    print(f"Random gradients spectral ratio: {ratio.item():.4f}")
    assert 1.0/batch_size <= ratio.item() <= 1.0

    # Test 2: Identical gradients (Consensus)
    # We can simulate this by passing the same x and y multiple times
    x_same = x[0:1].repeat(batch_size, 1)
    y_same = y[0:1].repeat(batch_size)
    ratio_consensus = get_spectrum_stats(params, model, x_same, y_same)
    print(f"Consensus gradients spectral ratio: {ratio_consensus.item():.4f}")
    # Should be close to 1.0
    assert ratio_consensus.item() > 0.99

    # Test 3: Differentiability
    ratio.backward()
    for name, p in params.items():
        assert p.grad is not None
        print(f"Gradient for {name} exists.")

    print("Logic verification successful!")

if __name__ == "__main__":
    test_logic()
