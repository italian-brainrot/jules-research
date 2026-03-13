import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call

def augment_data(x, shift_range=5, noise_std=0.01):
    """
    Apply random cyclic shift and additive Gaussian noise to 1D data.
    x: (batch_size, seq_len)
    """
    batch_size, seq_len = x.shape
    shifts = torch.randint(-shift_range, shift_range + 1, (batch_size,), device=x.device)

    # We use a vectorized way to apply cyclic shifts
    # For 1D data of length 40, we can use roll
    x_aug = torch.stack([torch.roll(x[i], shifts[i].item(), dims=0) for i in range(batch_size)])

    # Add noise
    noise = torch.randn_like(x_aug) * noise_std
    x_aug = x_aug + noise

    return x_aug

def compute_loss_single(params, buffers, model, x_single, y_single):
    # Ensure x_single has batch dimension for functional_call if model expects it
    # but here we use functional_call on a single sample, so we add dimension
    logits = functional_call(model, (params, buffers), (x_single.unsqueeze(0),))
    return F.cross_entropy(logits, y_single.unsqueeze(0))

def compute_aiga_loss(model, x, y, lambda_aiga=1.0):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    # Per-sample gradients for original data
    grad_fn = vmap(grad(compute_loss_single), in_dims=(None, None, None, 0, 0))
    grads_orig = grad_fn(params, buffers, model, x, y)

    # Augment data
    x_aug = augment_data(x)

    # Per-sample gradients for augmented data
    grads_aug = grad_fn(params, buffers, model, x_aug, y)

    # Flatten and concatenate gradients for each sample
    def flatten_grads(grads_dict):
        flat_grads = []
        # We need a consistent order, sorted by parameter name
        for name in sorted(grads_dict.keys()):
            g = grads_dict[name]
            flat_grads.append(g.reshape(x.shape[0], -1))
        return torch.cat(flat_grads, dim=1)

    flat_grads_orig = flatten_grads(grads_orig)
    flat_grads_aug = flatten_grads(grads_aug)

    # Cosine similarity
    # Normalize
    norm_orig = torch.norm(flat_grads_orig, p=2, dim=1, keepdim=True) + 1e-8
    norm_aug = torch.norm(flat_grads_aug, p=2, dim=1, keepdim=True) + 1e-8

    dir_orig = flat_grads_orig / norm_orig
    dir_aug = flat_grads_aug / norm_aug

    cos_sim = torch.sum(dir_orig * dir_aug, dim=1) # (batch_size,)

    # Loss is 1 - average cosine similarity
    aiga_penalty = 1.0 - torch.mean(cos_sim)

    return aiga_penalty

def test_augmentation():
    x = torch.randn(8, 40)
    x_aug = augment_data(x)
    assert x_aug.shape == x.shape
    assert not torch.equal(x, x_aug)
    print("test_augmentation passed")

def test_aiga_loss():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(40, 10)
        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    x = torch.randn(8, 40)
    y = torch.randint(0, 10, (8,))

    penalty = compute_aiga_loss(model, x, y)
    assert penalty >= 0
    assert penalty <= 2.0 # Cosine similarity is in [-1, 1], so 1-cos_sim is in [0, 2]
    print(f"test_aiga_loss passed, penalty: {penalty.item():.4f}")

if __name__ == "__main__":
    test_augmentation()
    test_aiga_loss()
