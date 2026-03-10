import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad

def compute_input_gradients(model, inputs, targets):
    def loss_fn(x, y):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        output = model(x)
        loss = F.cross_entropy(output, y)
        return loss

    # grad(loss_fn) w.r.t first argument (x)
    return vmap(grad(loss_fn))(inputs, targets)

def compute_icsd_penalty(input_gradients, targets, num_classes=10):
    # input_gradients: (B, 40)
    # targets: (B,)
    B, L = input_gradients.shape

    # 1. FFT to get spectral power
    # We use rfft for 1D real signals
    freqs = torch.fft.rfft(input_gradients, dim=1)
    power = torch.abs(freqs)**2 # (B, L//2 + 1)

    # 2. Normalize power spectrum to be a distribution
    power_norm = power / (power.sum(dim=1, keepdim=True) + 1e-8)

    # 3. Compute mean spectrum per class
    class_spectra = []
    valid_classes = []
    for c in range(num_classes):
        mask = (targets == c)
        if mask.any():
            class_mean = power_norm[mask].mean(dim=0)
            class_spectra.append(class_mean)
            valid_classes.append(c)

    if len(class_spectra) < 2:
        return torch.tensor(0.0, device=input_gradients.device)

    class_spectra = torch.stack(class_spectra) # (num_valid_classes, num_freqs)

    # 4. Compute divergence between classes
    # We want different classes to have DIFFERENT spectral signatures.
    # So we penalize Similarity between class-mean spectra.
    # Sim = mean_{i < j} cosine_similarity(S_i, S_j)

    # Pairwise cosine similarity
    dot_prod = torch.mm(class_spectra, class_spectra.t())
    norms = torch.norm(class_spectra, dim=1)
    norm_prod = torch.outer(norms, norms)
    cos_sim = dot_prod / (norm_prod + 1e-8)

    # Exclude self-similarity
    mask = torch.eye(len(valid_classes), device=input_gradients.device).bool()
    cos_sim_others = cos_sim[~mask]

    penalty = cos_sim_others.mean()
    return penalty

def test_logic():
    torch.manual_seed(42)
    B = 16
    L = 40
    num_classes = 4
    inputs = torch.randn(B, L, requires_grad=True)
    targets = torch.randint(0, num_classes, (B,))

    model = nn.Sequential(
        nn.Linear(L, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes)
    )

    input_grads = compute_input_gradients(model, inputs, targets)
    assert input_grads.shape == (B, L)
    print("Per-sample input gradients computed successfully.")

    penalty = compute_icsd_penalty(input_grads, targets, num_classes)
    print(f"ICSD Penalty: {penalty.item():.4f}")
    assert not torch.isnan(penalty)
    assert penalty >= 0

    # Test with identical gradients for different classes (should have high penalty)
    grads_identical = torch.ones(B, L)
    penalty_high = compute_icsd_penalty(grads_identical, targets, num_classes)
    print(f"Penalty for identical grads: {penalty_high.item():.4f}")
    assert penalty_high > 0.99

    # Test with orthogonal-ish spectra (should have lower penalty)
    grads_diverse = torch.zeros(B, L)
    for i in range(B):
        # Assign different frequencies to different samples/classes
        # This is a bit contrived but should show the point
        c = targets[i].item()
        grads_diverse[i, c % L] = 1.0

    # Note: real FFT of a spike is flat, so this won't give orthogonal spectra easily.
    # Let's just check that it's differentiable
    penalty.backward()
    print("Backward pass successful.")

if __name__ == "__main__":
    test_logic()
