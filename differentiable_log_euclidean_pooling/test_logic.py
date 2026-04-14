import torch
from model import LogEuclideanPooling

def test_dlep_logic():
    print("Testing LogEuclideanPooling logic...")
    batch_size = 2
    channels = 4
    length = 10

    x = torch.randn(batch_size, channels, length, requires_grad=True)
    layer = LogEuclideanPooling()

    out = layer(x)

    # Check output shape
    # C*(C+1)/2 = 4*5/2 = 10
    expected_dim = channels * (channels + 1) // 2
    assert out.shape == (batch_size, expected_dim), f"Expected shape {(batch_size, expected_dim)}, got {out.shape}"
    print(f"Output shape verified: {out.shape}")

    # Check differentiability
    loss = out.pow(2).sum()
    loss.backward()
    assert x.grad is not None, "Gradients should not be None"
    assert not torch.isnan(x.grad).any(), "Gradients should not contain NaNs"
    print("Differentiability verified.")

    # Check shift invariance (Log-Euclidean is NOT shift invariant because of covariance,
    # but we want to see it works for different inputs)
    x2 = x.detach().clone() + 1.0
    x2.requires_grad = True
    out2 = layer(x2)
    # Covariance is shift invariant!
    assert torch.allclose(out, out2, atol=1e-5), "DLEP should be shift invariant because covariance is computed on centered data"
    print("Shift invariance (via centering) verified.")

    # Scale check
    x3 = x.detach().clone() * 2.0
    out3 = layer(x3)
    # log(cov(2x)) = log(4 * cov(x)) = log(4) * I + log(cov(x))
    # This is not equal to out, which is expected.
    assert not torch.allclose(out, out3), "DLEP should not be scale invariant (expected)"
    print("Scale non-invariance verified (as expected for log-domain).")

if __name__ == "__main__":
    test_dlep_logic()
